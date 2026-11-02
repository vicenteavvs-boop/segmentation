import cv2
import numpy as np
from ultralytics import YOLO, SAM
from segment_anything import sam_model_registry, SamPredictor

yolo_model = YOLO("/home/fvalencia/nreyes/galaxy/runs/detect/trainX/weights/best.pt")
sam_model = SAM("sam2.1_l.pt")

img_path = "/home/fvalencia/nreyes/galaxy/imagenesprueba/prueba4.jpg"
image = cv2.imread(img_path)

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)

limg = cv2.merge((cl, a, b))
enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

results_yolo = yolo_model.predict(image, conf=0.25)
boxes = results_yolo[0].boxes.xyxy.cpu().numpy()

if len(boxes) > 0:
    all_points = []
    all_labels = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        margin = 0.05
        w, h = (x2 - x1), (y2 - y1)
        punto = [
            [cx, cy], 
            [x1 + w*margin, y1 + h*margin], 
            [x2 - w*margin, y1 + h*margin], 
            [x1 + w*margin, y2 - h*margin], 
            [x2 - w*margin, y2 - h*margin]
        ]
        all_points.append(punto)
        all_labels.append([1, 0, 0, 0, 0])

    results_sam = sam_model.predict(
        source=enhanced_img,
        bboxes=boxes,
        points=all_points,
        labels=all_labels,
        device="cpu"
    )

    results_sam[0].save("segmentado_clahe_final.jpg")
    cv2.imwrite("debug_clahe_input.jpg", enhanced_img) 
    print(f"Processed {len(boxes)} objects with CLAHE enhancement.")
