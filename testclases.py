from ultralytics import YOLO
model = YOLO("/home/fvalencia/nreyes/galaxy/runs/detect/trainX/weights/best.pt")
print(model.names)
