#!/usr/bin/env python
"""PrivacyVideo.py — Floutage sélectif de visages, + suivi CSRT

• Détection HOG + CSRT tracker pour combler les trous → plus de « clignotement ».
• Prévisualisation optionnelle (miniature 320 px toutes N frames).
• Tous les fichiers temporaires dans ./temp, sorties dans ./Output (output, output1…).
"""
# --------------------------------------------------------------------------- #
# Imports                                                                     #
# --------------------------------------------------------------------------- #
import os, shutil, tempfile
from pathlib import Path
from collections import Counter

import cv2
import face_recognition
import numpy as np
from tqdm import tqdm
import gradio as gr

# --------------------------------------------------------------------------- #
# Dossiers                                                                    #
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_ROOT  = SCRIPT_DIR / "temp";   TEMP_ROOT.mkdir(exist_ok=True)
OUTPUT_DIR = SCRIPT_DIR / "Output"; OUTPUT_DIR.mkdir(exist_ok=True)

os.environ["TMP"] = os.environ["TEMP"] = str(TEMP_ROOT)
tempfile.tempdir = str(TEMP_ROOT)

# --------------------------------------------------------------------------- #
# Utilitaires                                                                 #
# --------------------------------------------------------------------------- #

def next_output_path(stem="output", ext=".mp4") -> Path:
    i = 0
    while True:
        p = OUTPUT_DIR / f"{stem}{i if i else ''}{ext}"
        if not p.exists():
            return p
        i += 1


def blur_region(img, box, margin=0.0, k=99, sigma=30):
    t, r, b, l = box
    h, w = img.shape[:2]
    dx, dy = int((r-l)*margin), int((b-t)*margin)
    l, r = max(0,l-dx), min(w,r+dx)
    t, b = max(0,t-dy), min(h,b+dy)
    roi = img[t:b, l:r]
    if roi.size:
        img[t:b, l:r] = cv2.GaussianBlur(roi, (k,k), sigma)


def assign_id(enc, bank, thr):
    if not bank:
        return None
    d = face_recognition.face_distance([e for e,_ in bank], enc)
    idx = int(np.argmin(d))
    return bank[idx][1] if d[idx] < thr else None

# --------------------------------------------------------------------------- #
# Inventaire visages                                                          #
# --------------------------------------------------------------------------- #

def detect_faces(video_path, sample_rate, thr, max_thumbs=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Impossible d’ouvrir la vidéo.")

    bank, counts, thumbs = [], Counter(), {}
    next_id = 0
    tmp_dir = Path(tempfile.mkdtemp(prefix="faces_", dir=str(TEMP_ROOT)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for f in tqdm(range(0,total,sample_rate), desc="Inventaire"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read(); 0
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encs  = face_recognition.face_encodings(rgb, boxes)
        for box, enc in zip(boxes, encs):
            uid = assign_id(enc, bank, thr)
            if uid is None:
                uid = next_id; next_id +=1; bank.append((enc, uid))
            counts[uid]+=1
            if uid not in thumbs:
                t,r,b,l=box; cv2.imwrite(str(tmp_dir/f"id_{uid}.jpg"), frame[t:b,l:r]); thumbs[uid]=tmp_dir/f"id_{uid}.jpg"
    cap.release()

    displayed=[uid for uid,_ in counts.most_common(max_thumbs)]
    gallery=[[str(thumbs[uid]),f"id {uid}"] for uid in displayed]
    choices=[str(uid) for uid in displayed]
    ctx={"bank":[(e.tolist(),uid) for e,uid in bank],"video":video_path,"tmp_dir":str(tmp_dir)}
    return gallery, gr.update(choices=choices,value=[]), ctx

# --------------------------------------------------------------------------- #
# Traitement vidéo avec suivi CSRT                                            #
# --------------------------------------------------------------------------- #

def process_video(ctx, keep_ids, keep_mode, thr, margin, preview_on, preview_every):
    keep_ids = {int(x) for x in keep_ids}
    bank = [(np.array(e), uid) for e, uid in ctx["bank"]]

    cap = cv2.VideoCapture(ctx["video"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v"); fps=cap.get(cv2.CAP_PROP_FPS)
    w,h=int(cap.get(3)), int(cap.get(4))
    out_p = next_output_path(); writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w,h))

    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx=0

    trackers=[]  # list of dicts {trk, lost, blur}
    LOST_MAX=10

    for _ in tqdm(range(total), desc="Floutage"):
        ok, frame = cap.read(); 0
        if not ok: break

        # -- maj trackers existants --
        new_trackers=[]
        for tinfo in trackers:
            ok_box, bbox = tinfo['trk'].update(frame)
            if ok_box:
                tinfo['lost']=0
                x,y,w_box,h_box=map(int, bbox)
                blur_region(frame, (y, x+w_box, y+h_box, x), margin) if tinfo['blur'] else None
                new_trackers.append(tinfo)
            else:
                tinfo['lost']+=1
                if tinfo['lost']<LOST_MAX:
                    new_trackers.append(tinfo)
        trackers = new_trackers

        # -- détection visage sur frame courante --
        rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes=face_recognition.face_locations(rgb, model="hog")
        encs=face_recognition.face_encodings(rgb, boxes)
        for box,enc in zip(boxes,encs):
            uid = assign_id(enc, bank, thr)
            blur_flag = (uid not in keep_ids) if keep_mode else (uid in keep_ids)
            if blur_flag:
                blur_region(frame, box, margin)
            # créer tracker pour ce visage
            t,r,b,l = box
            trk = cv2.TrackerCSRT_create(); trk.init(frame, (l,t,r-l,b-t))
            trackers.append({'trk':trk, 'lost':0, 'blur':blur_flag})

        writer.write(frame)

        if preview_on and frame_idx % preview_every == 0:
            thumb=cv2.resize(frame, (320, int(h*320/w)))
            yield thumb, gr.update()
        frame_idx+=1

    cap.release(); writer.release(); shutil.rmtree(ctx["tmp_dir"], ignore_errors=True)
    for tmp in TEMP_ROOT.glob("wct*.tmp"): tmp.unlink(missing_ok=True)

    yield gr.update(), str(out_p)

# --------------------------------------------------------------------------- #
# Interface Gradio                                                            #
# --------------------------------------------------------------------------- #

def build_ui():
    with gr.Blocks(title="Floutage de visages") as demo:
        gr.Markdown("### Floutez les visages d’une vidéo.")
        with gr.Row():
            video_in  = gr.Video(label="Vidéo d’entrée (MP4…)")
            gallery   = gr.Gallery(label="Visages détectés", columns=[6])
        with gr.Accordion("Paramètres avancés", open=False):
            sample_r = gr.Slider(1,30,10,1,label="Échantillonnage (frames)")
            thr_s    = gr.Slider(0.35,0.80,0.48,0.01,label="Seuil de reconnaissance (thr)")
            margin_s = gr.Slider(0.0,0.30,0.0,0.01,label="Marge de flou (%)")
            preview_chk = gr.Checkbox(label="Prévisualiser (1 frame sur N)")
            preview_every = gr.Slider(1,60,15,1,label="N (frames)")
        keep_sel = gr.CheckboxGroup(label="Visages à garder nets (id)")
        detect_b = gr.Button("🔍 Détecter")
        keep_mode = gr.Checkbox(label="Tout flouter sauf sélection", value=True)
        process_b = gr.Button("🚀 Flouter vidéo")
        with gr.Row():
            preview_img = gr.Image(label="Prévisualisation", interactive=False)
            video_out = gr.Video(label="Vidéo floutée")
        ctx = gr.State()

        detect_b.click(detect_faces,
                       inputs=[video_in, sample_r, thr_s],
                       outputs=[gallery, keep_sel, ctx])
        process_b.click(process_video,
                       inputs=[ctx, keep_sel, keep_mode, thr_s, margin_s, preview_chk, preview_every],
                       outputs=[preview_img, video_out])
    return demo

if __name__ == "__main__":
    build_ui().launch(share=False, show_api=False)
