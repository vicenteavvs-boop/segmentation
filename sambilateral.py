import cv2
import numpy as np
from ultralytics import YOLO, SAM

yolo_model = YOLO("/home/fvalencia/nreyes/galaxy/runs/detect/trainX/weights/best.pt")
sam_model = SAM("sam2.1_l.pt")

img_path = "/home/fvalencia/nreyes/galaxy/imagenesprueba/prueba4.jpg"
image = cv2.imread(img_path)

results_yolo = yolo_model.predict(image, conf=0.25)
boxes = results_yolo[0].boxes.xyxy.cpu().numpy()

bilateral_img = cv2.bilateralFilter(image, 3, 1, 1)
cv2.imwrite('bilateral_blur.jpg', bilateral_img)

if len(boxes) > 0:
    all_points = []
    all_labels = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        margin = 0.05
        w, h = (x2 - x1), (y2 - y1)
        punto = [[cx, cy], [x1+w*margin, y1+h*margin], [x2-w*margin, y1+h*margin], 
                 [x1+w*margin, y2-h*margin], [x2-w*margin, y2-h*margin]]
        
        all_points.append(punto)
        all_labels.append([1, 0, 0, 0, 0])

    results_sam = sam_model.predict(
        source=bilateral_img, 
        bboxes=boxes,
        points=all_points,
        labels=all_labels,
        device="cpu"
    )

    results_sam[0].save("segmentado_bilateral_final.jpg")
    print("Segmentación completada sobre imagen con blur bilateral.")
