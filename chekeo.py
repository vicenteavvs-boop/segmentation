import os
import collections
from ultralytics import YOLO

model_path = "/home/fvalencia/nreyes/galaxy/runs/detect/trainX/weights/best.pt"
source_dir = "/home/fvalencia/nreyes/galaxy/COSMICA/test/images"

#chekeo 1
archivos = [f for f in os.listdir(source_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
c = collections.Counter(f.split(".rf.")[0] for f in archivos)
repetidos = {k: v for k, v in c.items() if v > 1}

print(f"Imagenes de origen: {len(archivos)}")
print(f"Nombres base repetidos: {len(repetidos)}")
for k, v in list(repetidos.items())[:20]:
    print(f"  {k}: {v} archivos")

#chekeo 2
model = YOLO(model_path)
results = model.predict(
    source=source_dir,
    save=False,
    conf=0.25,
    iou=0.5,
    agnostic_nms=True,
    verbose=False,
)

con_varias = 0
for r in results:
    n = len(r.boxes)
    if n > 1:
        con_varias += 1
        print(os.path.basename(r.path), n,
              [r.names[int(i)] for i in r.boxes.cls.tolist()],
              [round(x, 2) for x in r.boxes.conf.tolist()])

print(f"\nImagenes con más de una detección: {con_varias} de {len(results)}")
print(f"Nombres base repetidos : {len(repetidos)}")
