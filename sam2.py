from ultralytics import YOLO, SAM
import torch

yolo_model = YOLO("/home/fvalencia/nreyes/galaxy/runs/detect/trainX/weights/best.pt")
sam_model = SAM("sam2.1_l.pt")

img_path = "/home/fvalencia/nreyes/galaxy/opencv_binary_otsu.jpg"

results_yolo = yolo_model.predict(img_path, conf=0.25)
boxes = results_yolo[0].boxes.xyxy.cpu().numpy()

if len(boxes) > 0:
    all_points = []
    all_labels = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        margin = 0.05
        w, h = (x2 - x1), (y2 - y1)

        puntos_objeto = [
            [cx, cy], 
            [x1 + w*margin, y1 + h*margin], 
            [x2 - w*margin, y1 + h*margin], 
            [x1 + w*margin, y2 - h*margin], 
            [x2 - w*margin, y2 - h*margin]
        ]
        etiquetas_objeto = [1, 0, 0, 0, 0]
        
        all_points.append(puntos_objeto)
        all_labels.append(etiquetas_objeto)

    results_sam = sam_model.predict(
        img_path, 
        bboxes=boxes,      
        points=all_points, 
        labels=all_labels, 
        device="cpu"
    )


    results_sam[0].save("segmentado_final_completo.jpg")
    print(f"+++++ Éxito: Se han consolidado {len(boxes)} objetos en 'segmentado_final_completo.jpg'")

else:
    print("No se detectaron objetos para segmentar.")
