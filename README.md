a python script with a gradio interface to blur faces in videos (but you can choose faces not to blur) and maybe even license plates.
To run it in windows 10/11: To make it work: copy these files into a directory and then run "setup_and_run.bat"

un script en python avec une interface gradio pour flouter des visages sur les vidéos (mais on peut choisir des visages à ne pas flouter) et peut être même les plaques d'immatriculations.
Pour le faire fonctionner dans windows (10,11 ...): copiez ces fichiers dans un répertoire puis lancez "setup_and_run.bat"

<img width="654" alt="Sans titre" src="https://github.com/user-attachments/assets/bb4aecf2-134a-4d2b-a285-ff3be63b2e97" />

Réglages « catch-all » pour minimiser les visages manqués

Échantillonnage inventaire	1 frame	Chaque image est inspectée ; aucun visage n’est ignoré parce qu’il se trouve entre deux frames échantillonnées.

Seuil de reconnaissance (thr)	0.60 – 0.65	Seuil plus permissif : regroupe des poses/éclairages très variables et réduit les « ID fantômes ».

Marge de flou	0 (ou 0.05 si nécessaire)	Ne change rien à la détection, mais vous pouvez étendre légèrement le flou pour couvrir les bords.

Tout flouter sauf la sélection	Laisser coché	Ainsi les nouveaux visages apparus après inventaire sont floutés d’office, même s’ils n’ont pas d’ID.
