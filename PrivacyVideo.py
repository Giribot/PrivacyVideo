#!/usr/bin/env python
"""face_blur_app.py — visages & plaques (stockage local contrôlé)

• Tous les fichiers temporaires (vignettes, .tmp OpenCV, cache YOLO) vont
  dans   ./temp/
• Vidéos finales incrémentées dans   ./Output/
• Interface : sliders + checkbox plaques.
"""

# --------------------------------------------------------------------------- #
# Imports                                                                     #
# --------------------------------------------------------------------------- #
import os
import cv2, face_recognition, numpy as np, tempfile, shutil
from ultralytics import YOLO
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import gradio as gr

# --------------------------------------------------------------------------- #
# Répertoires & redirection des fichiers temporaires                          #
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_ROOT  = SCRIPT_DIR / "temp";   TEMP_ROOT.mkdir(exist_ok=True)
OUTPUT_DIR = SCRIPT_DIR / "Output"; OUTPUT_DIR.mkdir(exist_ok=True)

# Redirect Python/tempfile + OpenCV + Ultralytics cache to local folders
os.environ["TMP"] = os.environ["TEMP"] = str(TEMP_ROOT)
os.environ["ULTRALYTICS_HOME"] = str(SCRIPT_DIR / ".ultra")
# Force tempfile to use our directory
tempfile.tempdir = str(TEMP_ROOT)


def next_output_path(stem="output", ext=".mp4") -> Path:
    i = 0
    while True:
        p = OUTPUT_DIR / f"{stem}{i if i else ''}{ext}"
        if not p.exists():
            return p
        i += 1

# --------------------------------------------------------------------------- #
# Modèles                                                                     #
# --------------------------------------------------------------------------- #
try:
    PLATE_MODEL = YOLO("yolov8n-license_plate.pt")
except Exception as e:
    print("[!] YOLO licence-plate non chargé :", e)
    PLATE_MODEL = None

# --------------------------------------------------------------------------- #
# Utilitaires                                                                 #
# --------------------------------------------------------------------------- #

def blur_face(frame, box, k=99, sigma=30, margin=0.0):
    t, r, b, l = box
    h, w = frame.shape[:2]
    dx, dy = int((r - l) * margin), int((b - t) * margin)
    l, r = max(0, l - dx), min(w, r + dx)
    t, b = max(0, t - dy), min(h, b + dy)
    roi = frame[t:b, l:r]
    if roi.size:
        frame[t:b, l:r] = cv2.GaussianBlur(roi, (k, k), sigma)


def assign_id(enc, bank, thr):
    if not bank:
        return None
    dists = face_recognition.face_distance([e for e, _ in bank], enc)
    idx = int(np.argmin(dists))
    return bank[idx][1] if dists[idx] < thr else None

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

    for f in tqdm(range(0, total, sample_rate), desc="Inventaire"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, boxes)
        for box, enc in zip(boxes, encs):
            uid = assign_id(enc, bank, thr)
            if uid is None:
                uid = next_id; next_id += 1; bank.append((enc, uid))
            counts[uid] += 1
            if uid not in thumbs:
                t, r, b, l = box
                crop = frame[t:b, l:r]
                p = tmp_dir / f"id_{uid}.jpg"; cv2.imwrite(str(p), crop)
                thumbs[uid] = p
    cap.release()

    displayed = [uid for uid, _ in counts.most_common(max_thumbs)]
    gallery = [[str(thumbs[uid]), f"id {uid}"] for uid in displayed]
    choices = [str(uid) for uid in displayed]
    ctx = {"bank": [(e.tolist(), uid) for e, uid in bank],
           "video": video_path,
           "tmp_dir": str(tmp_dir)}
    return gallery, gr.update(choices=choices, value=[]), ctx

# --------------------------------------------------------------------------- #
# Floutage complet                                                            #
# --------------------------------------------------------------------------- #

def process_video(ctx, keep_ids_str, keep_mode, thr, margin, blur_plates):
    keep_ids = {int(x) for x in keep_ids_str}
    bank = [(np.array(e), uid) for e, uid in ctx["bank"]]

    cap = cv2.VideoCapture(ctx["video"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    out_path = next_output_path()
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total), desc="Floutage"):
        ok, frame = cap.read(); 0
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, boxes)
        for box, enc in zip(boxes, encs):
            uid = assign_id(enc, bank, thr)
            blur = (uid not in keep_ids) if keep_mode else (uid in keep_ids)
            if blur:
                blur_face(frame, box, margin=margin)

        if blur_plates and PLATE_MODEL is not None:
            results = PLATE_MODEL.predict(source=frame, conf=0.3, iou=0.5, verbose=False)
            for r in results:
                for xyxy in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, xyxy)
                    blur_face(frame, (y1, x2, y2, x1), margin=0.05)
        writer.write(frame)

    cap.release(); writer.release(); shutil.rmtree(ctx["tmp_dir"], ignore_errors=True)

    # Nettoyage éventuel des .tmp OpenCV restants
    for tmp in TEMP_ROOT.glob("wct*.tmp"):
        try:
            tmp.unlink()
        except PermissionError:
            pass

    return str(out_path)

# --------------------------------------------------------------------------- #
# Interface Gradio                                                            #
# --------------------------------------------------------------------------- #

def build_ui():
    with gr.Blocks(title="Floutage de visages & plaques") as demo:
        gr.Markdown("### Floutez visages et plaques d’immatriculation d’une vidéo.")
        with gr.Row():
            video_input = gr.Video(label="Vidéo d’entrée (MP4…)")
            gallery = gr.Gallery(label="Visages détectés", columns=[6])
        with gr.Accordion("Paramètres avancés", open=False):
            sample_rate_slider = gr.Slider(1, 30, 10, 1, label="Échantillonnage inventaire (frames)",
                                          info="Analyse 1 frame sur N durant l’inventaire.")
            thr_slider = gr.Slider(0.35, 0.80, 0.48, 0.01, label="Seuil de reconnaissance (thr)",
                                   info="0.35 strict → 0.80 permissif : fusion des poses d'un même visage.")
            margin_slider = gr.Slider(0.0, 0.30, 0.0, 0.01, label="Marge de flou (%)",
                                      info="Agrandit la zone floutée autour du visage.")
        with gr.Row():
            keep_select = gr.CheckboxGroup(label="Visages à garder nets (id)")
            detect_btn = gr.Button("🔍 Déetecter les visages")
        keep_mode = gr.Checkbox(label="Tout flouter sauf la sélection", value=True,
                              info="Coché : seuls les visages sélectionnés resteront nets.")
        blur_plates_chk = gr.Checkbox(label="Flouter les plaques d'immatriculation", value=False,
                                      info="Utilise YOLOv8 licence-plate. Plus lent sur CPU.")
        with gr.Row():
            process_btn = gr.Button("🚀 Lancer le floutage")
            output
