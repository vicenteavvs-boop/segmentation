import cv2
import numpy as np
import torch
import glob
import os
from segment_anything import sam_model_registry, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
sam_hq = sam_model_registry["vit_l"](checkpoint="sam_hq_vit_l.pth")
sam_hq.to(device=device)
predictor = SamPredictor(sam_hq)

crops_dir = '/home/fvalencia/nreyes/galaxy/runs/detect/predict/crops'
preprocesado_dir = '/home/fvalencia/nreyes/galaxy/preprocesado'
output_root = '/home/fvalencia/nreyes/galaxy/segmentado'

preprocesado_paths = sorted(glob.glob(os.path.join(preprocesado_dir,'*', 'input_para_sam_*.jpg')))
print(f"Se encontraron {len(preprocesado_paths)} imagenes para segmentar")

MAX_AREA_RATIO = 0.90
box_margin = 0.05

for preproc_path in preprocesado_paths:
    clase = os.path.basename(os.path.dirname(preproc_path))
    nombre = os.path.basename(preproc_path).replace('input_para_sam_', '').replace('.jpg', '')
    crop_path = os.path.join(crops_dir, f"{nombre}.jpg")

    im = cv2.imread(preproc_path)
    if im is None:
        print(f"Aviso: no se pudo leer {preproc_path}, se omite")
        continue

    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    predictor.set_image(im_rgb)

    h, w = im.shape[:2]

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

    candidatas = sorted(zip(masks, scores), key=lambda x: x[1], reverse=True)
    mask, mask_score = None, None
    for m, s in candidatas:
        area_ratio = m.sum() / m.size
        if area_ratio <= MAX_AREA_RATIO:
            mask, mask_score = m, s
            break

    if mask is None:
        mask, mask_score = candidatas[0]
        print(f"[{clase}/{nombre}] Aviso: todas las máscaras cubren más del {MAX_AREA_RATIO:.0%} "
              f"del área; se usó la de mayor score de todas formas (score={mask_score:.3f}).")
    else:
        print(f"[{clase}/{nombre}] Máscara seleccionada -> score={mask_score:.3f}, "
              f"área cubierta={mask.sum() / mask.size:.1%}")

    final_viz = cv2.imread(crop_path) if os.path.exists(crop_path) else im.copy()
    final_viz[mask] = final_viz[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

    out_dir = os.path.join(output_root, clase)
    os.makedirs(out_dir, exist_ok = True)
    out_path = os.path.join(out_dir, f"resultado_final_{nombre}.jpg")
    cv2.imwrite(out_path, final_viz)

print(f"Segmentacion lista ({len(preprocesado_paths)} imagenes procesadas)")
