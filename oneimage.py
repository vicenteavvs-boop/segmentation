from ultralytics import YOLO

model_path="/home/fvalencia/nreyes/galaxy/runs/detect/trainX/weights/best.pt"
model = YOLO(model_path)

results = model.predict(
	source="/home/fvalencia/nreyes/galaxy/COSMICA/test/images/m13-s_jpg.rf.a99355f96f9faf9cc881ca4d6ac79e88.jpg",
	save=True,
	save_crop=True,
	conf=0.25
)

print(f"Foto localizada guardada en: {results[0].save_dir}")
