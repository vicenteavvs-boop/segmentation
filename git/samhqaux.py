import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_l"
checkpoint = "sam_hq_vit_l.pth"
sam_hq = sam_model_registry[model_type](checkpoint=checkpoint)
sam_hq.to(device=device)
predictor = SamPredictor(sam_hq)

crop_to_segment = cv2.imread('/home/fvalencia/nreyes/galaxy/runs/detect/predict3/crops/globular_cluster/m13-s_jpg.jpg')
crop_rgb = cv2.cvtColor(crop_to_segment, cv2.COLOR_BGR2RGB)
predictor.set_image(crop_rgb)

h, w = crop_to_segment.shape[:2]
input_point = np.array([[w // 2, h // 2]]) 
input_label = np.array([1])

input_box = np.array([0, 0, w, h])

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False
)

mask = masks[0]
res_viz = crop_to_segment.copy()
res_viz[mask] = res_viz[mask] * 0.5 + np.array([0, 255, 0]) * 0.5 

cv2.imwrite('resultadosam.jpg' ,res_viz)
print("segmentacion directa en 'resultadosam.jpg' guardado")
