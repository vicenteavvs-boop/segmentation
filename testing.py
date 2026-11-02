import os
from ultralytics import YOLO

model_path ="/home/fvalencia/nreyes/galaxy/runs/detect/trainN/weights/best.pt"
model = YOLO(model_path)

results = model.val(
	data = "dataset.yaml",
	split = 'test',
	save_json=True,
)

save_dir = results.save_dir
report_file = os.path.join(save_dir, "metricas.txt")

with open(report_file, "w") as f:
	f.write(f"Resultados para {model_path}\n\n")
	f.write(f"Promedio mAP50: {results.box.map50:.4f}\n")
	f.write(f"Promedio mAP50-95: {results.box.map:4f}\n")
	f.write(f"Precision: {results.box.mp:.4f}\n")
	f.write(f"Recall: {results.box.mr:.4f}\n\n")

	f.write("mAP por clase:\n")
	for i,name in results.names.items():
		m50 = results.box.ap50[i]
		m50_95 = results.box.maps[i]
		f.write(f" - {name:23}: {m50:.4} | {m50_95:.4f}\n")

print(f"Reporte guardado en: {report_file}")
