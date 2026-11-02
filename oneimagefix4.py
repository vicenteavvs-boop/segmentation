import os
import json
import shutil
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None

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

tam = {}
for f in archivos:
    try:
        with Image.open(os.path.join(source_dir, f)) as im_:
            tam[f] = im_.size
    except Exception:
        tam[f] = (0, 0)

print("Las 5 fotos más grandes (px):")
for f, (w_, h_) in sorted(tam.items(), key=lambda kv: kv[1][0] * kv[1][1], reverse=True)[:5]:
    print(f"  {f}: {w_}x{h_} ({w_ * h_ / 1e6:.0f} MP)")

n_imgs, n_crops = 0, 0
omitidas = []

for f in archivos:
    path = os.path.join(source_dir, f)
    try:
        r = model.predict(
            source=path,
            save=True,         
            conf=0.25,
            iou=0.5,
            batch=1,
            verbose=False,
            project="/home/fvalencia/nreyes/galaxy/runs/detect",
            name="predict",
            exist_ok=True,
            agnostic_nms=True,
        )[0]
    except torch.cuda.OutOfMemoryError:
        w_, h_ = tam.get(f, (0, 0))
        print(f"Sin memoria GPU en {f} ({w_}x{h_}); se omite")
        torch.cuda.empty_cache()
        omitidas.append(f)
        continue

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
        os.makedirs(clase_dir, exist_ok=sTrue)

        nombre_crop = f"{stem}_d{k}"
        cv2.imwrite(os.path.join(clase_dir, f"{nombre_crop}.jpg"), r.orig_img[y1:y2, x1:x2])

        with open(os.path.join(clase_dir, f"{nombre_crop}.json"), "w") as jf:
            json.dump({"source": r.path, "box": [x1, y1, x2, y2],
                       "clase": clase, "conf": float(box.conf[0])}, jf)
        n_crops += 1

print(f"Se procesaron {n_imgs} imagenes, {n_crops} crops guardados en {crops_root}")
if omitidas:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "omitidas_sin_memoria.txt"), "w") as of:
        of.write("\n".join(omitidas))
    print(f"{len(omitidas)} fotos omitidas por memoria GPU (lista en {out_dir}/omitidas_sin_memoria.txt)")
