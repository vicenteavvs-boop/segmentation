from ultralytics import YOLO, SAM
import torch

yolo_model = YOLO("/home/fvalencia/nreyes/galaxy/runs/detect/trainX/weights/best.pt")

sam_model = SAM("sam2.1_l.pt") 

img_path = "/home/fvalencia/nreyes/galaxy/imagenesprueba/prueba4.jpg"
results_yolo = yolo_model.predict(img_path, conf=0.25)


boxes = results_yolo[0].boxes.xyxy.cpu().numpy()

if len(boxes) > 0:
    for i, box in enumerate(boxes):
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
        
       
        etiqueta = [1, 0, 0, 0, 0]

        results_sam = sam_model.predict(
            img_path, 
            bboxes=[box],
            points=[punto],
            labels=[etiqueta],
        )

        results_sam[0].save(f"segmentado_hq_{i}.jpg")
        print(f"+++++ Cúmulo/Galaxia {i} procesada con HQ-SAM")
else:
    print("nop ------ No se detectaron objetos")
