import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
sam_hq = sam_model_registry["vit_l"](checkpoint="sam_hq_vit_l.pth")
sam_hq.to(device=device)
predictor = SamPredictor(sam_hq)

path = ('/home/fvalencia/nreyes/galaxy/runs/detect/predict3/crops/globular_cluster/m13-s_jpg.jpg')

im = cv2.imread('input_para_sam.jpg')
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
predictor.set_image(im_rgb)

h, w = im.shape[:2]
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
final_viz = cv2.imread(path)
final_viz[mask] = final_viz[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

cv2.imwrite('resultado_final_segmentado.jpg', final_viz)
print("Segmentacion lista")
