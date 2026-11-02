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

box_margin = 0.05
input_box = np.array([
    w * box_margin, h * box_margin,
    w * (1 - box_margin), h * (1 - box_margin)
])


input_point = np.array([
    [w // 2, h // 2],
    [w * 0.02, h * 0.02],
    [w * 0.98, h * 0.02],
    [w * 0.02, h * 0.98],
    [w * 0.98, h * 0.98],
])
input_label = np.array([1, 0, 0, 0, 0])

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=True
)


MAX_AREA_RATIO = 0.90
candidatas = sorted(zip(masks, scores), key=lambda x: x[1], reverse=True)

mask, mask_score = None, None
for m, s in candidatas:
    area_ratio = m.sum() / m.size
    if area_ratio <= MAX_AREA_RATIO:
        mask, mask_score = m, s
        break

if mask is None:
    mask, mask_score = candidatas[0]
    print(f"Aviso: todas las máscaras cubren más del {MAX_AREA_RATIO:.0%} del área; "
          f"se usó la de mayor score de todas formas (score={mask_score:.3f}).")
else:
    print(f"Máscara seleccionada -> score={mask_score:.3f}, "
          f"área cubierta={mask.sum() / mask.size:.1%}")

final_viz = cv2.imread(path)
final_viz[mask] = final_viz[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

cv2.imwrite('resultado_final_segmentado.jpg', final_viz)
print("Segmentacion lista")
