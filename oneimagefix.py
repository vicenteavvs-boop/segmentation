from ultralytics import YOLO

model_path = "/home/fvalencia/nreyes/galaxy/runs/detect/trainX/weights/best.pt"
model = YOLO(model_path)

source_dir = "/home/fvalencia/nreyes/galaxy/COSMICA/test/images"

results = model.predict(
    source=source_dir,
    save=True,
    save_crop=True,
    conf=0.25,
    iou=0.5,
    project="/home/fvalencia/nreyes/galaxy/runs/detect",
    name="predict",
    exist_ok=True,
    agnostic_nms=True   
)

print(f"Se procesaron {len(results)} imagenes")
print(f"Resultados guardados en: {results[0].save_dir}")
