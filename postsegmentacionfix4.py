import cv2
import numpy as np
import torch
import glob
import os
import json
import shutil
from segment_anything import sam_model_registry, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
sam_hq = sam_model_registry["vit_l"](checkpoint="sam_hq_vit_l.pth")
sam_hq.to(device=device)
predictor = SamPredictor(sam_hq)

crops_root = '/home/fvalencia/nreyes/galaxy/runs/detect/predict/crops'
preprocesado_root = '/home/fvalencia/nreyes/galaxy/preprocesado'
output_root = '/home/fvalencia/nreyes/galaxy/segmentado'

MAX_AREA_RATIO = 0.90
box_margin = 0.05

if os.path.exists(output_root):
    shutil.rmtree(output_root)

preprocesado_paths = sorted(glob.glob(os.path.join(preprocesado_root, '*', 'input_para_sam_*.jpg')))
print(f"Se encontraron {len(preprocesado_paths)} imagenes para segmentar")

for preproc_path in preprocesado_paths:
    clase = os.path.basename(os.path.dirname(preproc_path))
    nombre = os.path.basename(preproc_path).replace('input_para_sam_', '').replace('.jpg', '')
    crop_path = os.path.join(crops_root, clase, f"{nombre}.jpg")
    json_path = preproc_path.replace('.jpg', '.json')

    im = cv2.imread(preproc_path)
    if im is None:
        print(f"Aviso: no se pudo leer {preproc_path}, se omite")
        continue

    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    predictor.set_image(im_rgb)
    h, w = im.shape[:2]

    if os.path.exists(json_path):
        with open(json_path) as f:
            sidecar = json.load(f)
        otsu_centroid = np.array(sidecar["centroid"])
        positive_points = sidecar["positive_points"]
        negative_points = sidecar["negative_points"]
        if not sidecar["otsu_confiable"]:
            print(f"[{clase}/{nombre}] Aviso: Otsu no encontró un objeto confiable en el preprocesamiento; "
                  f"la segmentación puede no ser fiable.")
    else:
        
        print(f"[{clase}/{nombre}] Aviso: no se encontró {os.path.basename(json_path)}, usando centro geométrico")
        otsu_centroid = np.array([w / 2, h / 2])
        positive_points = [[w / 2, h / 2]]
        negative_points = [[w * 0.02, h * 0.02], [w * 0.98, h * 0.02],
                            [w * 0.02, h * 0.98], [w * 0.98, h * 0.98]]

    input_point = np.array(positive_points + negative_points)
    input_label = np.array([1] * len(positive_points) + [0] * len(negative_points))

    input_box = np.array([
        w * box_margin, h * box_margin,
        w * (1 - box_margin), h * (1 - box_margin)
    ])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=True
    )

   
    dist_umbral = 0.6 * max(w, h)
    candidatas = sorted(zip(masks, scores), key=lambda x: x[1], reverse=True)

    def centroide_mascara(m):
        ys, xs = np.where(m)
        if len(xs) == 0:
            return None
        return np.array([xs.mean(), ys.mean()])

    mask, mask_score, motivo = None, None, None
    for m, s in candidatas:
        area_ratio = m.sum() / m.size
        if area_ratio > MAX_AREA_RATIO:
            continue
        c = centroide_mascara(m)
        if c is not None and np.linalg.norm(c - otsu_centroid) <= dist_umbral:
            mask, mask_score, motivo = m, s, "área y centroide OK"
            break

    if mask is None:
        for m, s in candidatas:
            area_ratio = m.sum() / m.size
            if area_ratio <= MAX_AREA_RATIO:
                mask, mask_score, motivo = m, s, "solo área OK (centroide lejos del objeto esperado)"
                break

    if mask is None:
        mask, mask_score = candidatas[0]
        motivo = "ninguna candidata pasó los filtros; se usó la de mayor score"

    print(f"[{clase}/{nombre}] Máscara seleccionada -> score={mask_score:.3f}, "
          f"área={mask.sum() / mask.size:.1%} ({motivo})")

    final_viz = cv2.imread(crop_path) if os.path.exists(crop_path) else im.copy()
    final_viz[mask] = final_viz[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

    out_dir = os.path.join(output_root, clase)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resultado_final_{nombre}.jpg")
    cv2.imwrite(out_path, final_viz)

print(f"Segmentacion lista ({len(preprocesado_paths)} imagenes procesadas)")
