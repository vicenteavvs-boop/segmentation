import cv2
import numpy as np
import torch
import glob
import os
import json
import collections
import shutil
from segment_anything import sam_model_registry, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
sam_hq = sam_model_registry["vit_l"](checkpoint="sam_hq_vit_l.pth")
sam_hq.to(device=device)
predictor = SamPredictor(sam_hq)

crops_root = '/home/fvalencia/nreyes/galaxy/runs/detect/predict/crops'
preprocesado_root = '/home/fvalencia/nreyes/galaxy/preprocesado'
output_root = '/home/fvalencia/nreyes/galaxy/segmentado'
masks_dir = os.path.join(output_root, '_masks')  

MAX_AREA_RATIO = 0.90
box_margin = 0.05
MAX_LADO = 2048  

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

    h0, w0 = im.shape[:2]
    esc = min(1.0, MAX_LADO / max(h0, w0))
    if esc < 1.0:
        im = cv2.resize(im, (int(round(w0 * esc)), int(round(h0 * esc))), interpolation=cv2.INTER_AREA)
    h, w = im.shape[:2]

    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    try:
        predictor.set_image(im_rgb)
    except torch.cuda.OutOfMemoryError:
        print(f"[{clase}/{nombre}] Sin memoria GPU en set_image (crop {w0}x{h0}); se omite")
        torch.cuda.empty_cache()
        continue

    if os.path.exists(json_path):
        with open(json_path) as f:
            sidecar = json.load(f)
        otsu_centroid = np.array(sidecar["centroid"]) * esc  
        positive_points = (np.array(sidecar["positive_points"]) * esc).tolist()
        negative_points = (np.array(sidecar["negative_points"]) * esc).tolist()
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

    try:
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True
        )
    except torch.cuda.OutOfMemoryError:
        print(f"[{clase}/{nombre}] Sin memoria GPU en predict (crop {w0}x{h0}, enviado como {w}x{h}); se omite")
        torch.cuda.empty_cache()
        continue

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

    if esc < 1.0:
        mask = cv2.resize(mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST) > 0

    final_viz = cv2.imread(crop_path) if os.path.exists(crop_path) else cv2.resize(im, (w0, h0))
    final_viz[mask] = final_viz[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

    out_dir = os.path.join(output_root, clase)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resultado_final_{nombre}.jpg")
    cv2.imwrite(out_path, final_viz)

    os.makedirs(os.path.join(masks_dir, clase), exist_ok=True)
    cv2.imwrite(os.path.join(masks_dir, clase, f"{nombre}.png"), mask.astype(np.uint8) * 255)

completas_dir = os.path.join(output_root, "completas")
os.makedirs(completas_dir, exist_ok=True)

clases = sorted(d for d in os.listdir(crops_root) if os.path.isdir(os.path.join(crops_root, d)))
paleta = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]  # BGR
color_clase = {c: paleta[i % len(paleta)] for i, c in enumerate(clases)}

por_origen = collections.defaultdict(list)
for meta_path in glob.glob(os.path.join(crops_root, '*', '*.json')):
    with open(meta_path) as f:
        meta = json.load(f)
    por_origen[meta["source"]].append((os.path.splitext(os.path.basename(meta_path))[0], meta))

n_completas = 0
for source, items in por_origen.items():
    orig = cv2.imread(source)
    if orig is None:
        print(f"Aviso: no se pudo leer la foto original {source}, se omite")
        continue

    grosor = max(1, int(round(max(orig.shape[:2]) / 600)))
    pintadas = 0
    for nombre, meta in items:
        mask_path = os.path.join(masks_dir, meta["clase"], f"{nombre}.png")
        if not os.path.exists(mask_path):
            continue 
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
        x1, y1, x2, y2 = meta["box"]
        if m.shape != (y2 - y1, x2 - x1):
            m = cv2.resize(m.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) > 0

        color = color_clase[meta["clase"]]
        region = orig[y1:y2, x1:x2]  
        region[m] = (region[m] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        cv2.rectangle(orig, (x1, y1), (x2, y2), color, grosor)
        pintadas += 1

    if pintadas:
        stem = os.path.splitext(os.path.basename(source))[0].replace(".", "_")
        cv2.imwrite(os.path.join(completas_dir, f"{stem}.jpg"), orig)
        n_completas += 1

print(f"Fase 2: {n_completas} fotos completas con máscaras en {completas_dir}")
print(f"Segmentacion lista ({len(preprocesado_paths)} imagenes procesadas)")
