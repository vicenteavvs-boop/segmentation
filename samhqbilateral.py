import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

yolo_model = YOLO("/home/fvalencia/nreyes/galaxy/runs/detect/trainX/weights/best.pt")


device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_l" 
checkpoint = "sam_hq_vit_l.pth" 

sam_hq = sam_model_registry[model_type](checkpoint=checkpoint)
sam_hq.to(device=device)
predictor = SamPredictor(sam_hq)


img_path = "/home/fvalencia/nreyes/galaxy/imagenesprueba/prueba4.jpg"
image = cv2.imread(img_path)

bilateral_image = cv2.bilateralFilter(image, 3, 1, 1)

bilateral_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(bilateral_image_rgb)


results_yolo = yolo_model.predict(image, conf=0.25)
boxes = results_yolo[0].boxes.xyxy.cpu().numpy()


if len(boxes) > 0:
    
    final_mask_viz = bilateral_image.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
      
        margin = 0.05
        w, h = (x2 - x1), (y2 - y1)
        
       
        input_points = np.array([
            [cx, cy], 
            [x1 + w*margin, y1 + h*margin], 
            [x2 - w*margin, y1 + h*margin], 
            [x1 + w*margin, y2 - h*margin], 
            [x2 - w*margin, y2 - h*margin]
        ])
        input_labels = np.array([1, 0, 0, 0, 0])

        
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=box,
            multimask_output=False
        )

        
        mask = masks[0]
        color = np.random.randint(0, 255, (3,)).tolist()
        
        final_mask_viz[mask] = final_mask_viz[mask] * 0.5 + np.array(color) * 0.5
   
    cv2.imwrite("segmentado_hq_bilateral.jpg", final_mask_viz)
    cv2.imwrite("bilateral.jpg", filtro_bilateral)
    print(f"HQ-SAM procesó {len(boxes)} objetos con éxito.")

