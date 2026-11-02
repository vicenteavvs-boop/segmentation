import os
import json
import shutil
import cv2
from ultralytics import YOLO
 
model_path = "/home/fvalencia/nreyes/galaxy/runs/detect/trainX/weights/best.pt"
model = YOLO(model_path)
 
source_dir = "/home/fvalencia/nreyes/galaxy/COSMICA/test/images"
out_dir = "/home/fvalencia/nreyes/galaxy/runs/detect/predict"
crops_root = os.path.join(out_dir, "crops")
 
UNA_POR_NOMBRE_BASE = False
 
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
 
archivos = sorted(f for f in os.listdir(source_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
 
if UNA_POR_NOMBRE_BASE:
    vistos, filtrados = set(), []
    for f in archivos:
        base = f.split(".rf.")[0]
        if base not in vistos:
            vistos.add(base)
            filtrados.append(f)
    print(f"{len(archivos) - len(filtrados)} archivos omitidos por nombre base repetido")
    archivos = filtrados
 
sources = [os.path.join(source_dir, f) for f in archivos]

results = model.predict(
    source=sources,
    stream=True,
    save=True,         
    conf=0.25,
    iou=0.5,
    project="/home/fvalencia/nreyes/galaxy/runs/detect",
    name="predict",
    exist_ok=True,
    agnostic_nms=True,
)
 
n_imgs, n_crops = 0, 0
for r in results:
    n_imgs += 1
    stem = os.path.splitext(os.path.basename(r.path))[0].replace(".", "_")
    h, w = r.orig_img.shape[:2]
 
    for k, box in enumerate(r.boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            continue
 
        clase = r.names[int(box.cls[0])]
        clase_dir = os.path.join(crops_root, clase)
        os.makedirs(clase_dir, exist_ok=True)
 
        nombre_crop = f"{stem}_d{k}"
        cv2.imwrite(os.path.join(clase_dir, f"{nombre_crop}.jpg"), r.orig_img[y1:y2, x1:x2])
 
        with open(os.path.join(clase_dir, f"{nombre_crop}.json"), "w") as f:
            json.dump({"source": r.path, "box": [x1, y1, x2, y2],
                       "clase": clase, "conf": float(box.conf[0])}, f)
        n_crops += 1
 
print(f"Se procesaron {n_imgs} imagenes, {n_crops} crops guardados en {crops_root}")
