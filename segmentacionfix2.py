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
revisar_dir = os.path.join(output_root, '_revisar')  

MAX_LADO = 2048 

USAR_CAJA_OTSU = True
BBOX_PAD = 0.10        
MIN_BBOX_FRAC = 0.10   
box_margin = 0.05      

CLASES_REDONDAS = {"globular_cluster", "galaxy"}

MIN_AREA_RATIO = 0.005  
MAX_AREA_RATIO = 0.90   
MAX_BORDE = 0.70        
MAX_EXTENT = 0.93       
REINTENTAR_CON_CROP_ORIGINAL = True  
EXCLUIR_SOSPECHOSAS = True      

KERNEL_FRAC = 0.01  
LIMPIEZA_DEFAULT = {"apertura": True, "cierre": True, "componente": True, "rellenar": True}
LIMPIEZA_POR_CLASE = {
    "comet":  {"apertura": False, "rellenar": False},  
    "nebula": {"apertura": False, "rellenar": False},  
}


def limpiar_mascara(mask, punto, cfg):
    """Devuelve la máscara (bool) limpiada según cfg. Si la limpieza la dejara vacía,
    devuelve la original sin cambios."""
    m = mask.astype(np.uint8)
    h_, w_ = m.shape
    k = max(3, int(round(min(h_, w_) * KERNEL_FRAC)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if cfg["apertura"]:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    if cfg["cierre"]:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    if cfg["componente"]:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            x = int(np.clip(round(punto[0]), 0, w_ - 1))
            y = int(np.clip(round(punto[1]), 0, h_ - 1))
            lab = labels[y, x]
            if lab == 0:  
                lab = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (labels == lab).astype(np.uint8)

    if cfg["rellenar"]:
        contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        relleno = np.zeros_like(m)
        cv2.drawContours(relleno, contornos, -1, 1, thickness=cv2.FILLED)
        m = relleno

    return (m > 0) if m.any() else mask


def metricas_mascara(m):
    """(área relativa, extent, fracción del borde del crop cubierta)."""
    area = float(m.sum())
    area_ratio = area / m.size
    ys, xs = np.where(m)
    if len(xs) == 0:
        return area_ratio, 0.0, 0.0
    bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    extent = area / bbox_area
    borde = np.concatenate([m[0, :], m[-1, :], m[:, 0], m[:, -1]]).mean()
    return area_ratio, float(extent), float(borde)


def motivos_sospecha(m, clase):
    """Lista de motivos por los que la máscara parece incorrecta (vacía = parece buena)."""
    area_ratio, extent, borde = metricas_mascara(m)
    motivos = []
    if area_ratio < MIN_AREA_RATIO:
        motivos.append(f"área casi nula ({area_ratio:.1%})")
    if area_ratio > MAX_AREA_RATIO:
        motivos.append(f"área {area_ratio:.0%}")
    if borde > MAX_BORDE:
        motivos.append(f"cubre {borde:.0%} del borde")
    if clase in CLASES_REDONDAS and extent > MAX_EXTENT:
        motivos.append(f"forma rectangular (extent {extent:.2f})")
    return motivos


def centroide_mascara(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def calcular_caja(sidecar, w, h, esc):
    """Caja (x1, y1, x2, y2) para SAM en las coordenadas del crop ya escalado."""
    por_defecto = np.array([w * box_margin, h * box_margin, w * (1 - box_margin), h * (1 - box_margin)])
    if not USAR_CAJA_OTSU or not sidecar or not sidecar.get("otsu_confiable") or not sidecar.get("otsu_bbox"):
        return por_defecto, "caja por defecto"

    x1, y1, x2, y2 = [v * esc for v in sidecar["otsu_bbox"]]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0 or bw * bh < MIN_BBOX_FRAC * w * h:
        return por_defecto, "caja por defecto (la de Otsu es muy chica)"

    px, py = bw * BBOX_PAD, bh * BBOX_PAD
    caja = np.array([max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)])
    return caja, "caja ajustada a Otsu"


def esquinas_fondo(im, w, h):
    """Puntos cerca de las 4 esquinas que Otsu marcó como fondo (parche negro en la imagen preprocesada)."""
    puntos = []
    for fx, fy in [(0.02, 0.02), (0.98, 0.02), (0.02, 0.98), (0.98, 0.98)]:
        x, y = int(round(fx * (w - 1))), int(round(fy * (h - 1)))
        parche = im[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        if parche.max() == 0:
            puntos.append([x, y])
    return puntos


def segmentar(img_bgr, input_point, input_label, input_box, otsu_centroid, clase, punto, cfg):
    """Corre SAM-HQ sobre img_bgr, elige la mejor candidata y la limpia.
    Devuelve (máscara, score, motivo de la elección, área antes de limpiar)."""
    h_, w_ = img_bgr.shape[:2]
    predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=True
    )

    dist_umbral = 0.6 * max(w_, h_)
    candidatas = sorted(zip(masks, scores), key=lambda x: x[1], reverse=True)

    mask, mask_score, motivo = None, None, None
    for m, s in candidatas:
        if motivos_sospecha(m, clase):
            continue
        c = centroide_mascara(m)
        if c is not None and np.linalg.norm(c - otsu_centroid) <= dist_umbral:
            mask, mask_score, motivo = m, s, "forma y centroide OK"
            break

    if mask is None:
        for m, s in candidatas:
            if not motivos_sospecha(m, clase):
                mask, mask_score, motivo = m, s, "solo forma OK (centroide lejos del objeto esperado)"
                break

    if mask is None:
        mask, mask_score = candidatas[0]
        motivo = "ninguna candidata pasó los filtros; se usó la de mayor score"

    area_antes = mask.sum() / mask.size
    mask = limpiar_mascara(mask, punto, cfg)
    return mask, float(mask_score), motivo, area_antes

if os.path.exists(output_root):
    shutil.rmtree(output_root)

preprocesado_paths = sorted(glob.glob(os.path.join(preprocesado_root, '*', 'input_para_sam_*.jpg')))
print(f"Se encontraron {len(preprocesado_paths)} imagenes para segmentar")

revisar = []

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


    sidecar = None
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

    n_esquinas = 0
    if clase in CLASES_REDONDAS:
        esquinas = esquinas_fondo(im, w, h)
        negative_points = negative_points + esquinas
        n_esquinas = len(esquinas)

    input_point = np.array(positive_points + negative_points)
    input_label = np.array([1] * len(positive_points) + [0] * len(negative_points))
    input_box, tipo_caja = calcular_caja(sidecar, w, h, esc)

    cfg = {**LIMPIEZA_DEFAULT, **LIMPIEZA_POR_CLASE.get(clase, {})}
    punto = positive_points[0]

    try:
        mask, mask_score, motivo, area_antes = segmentar(
            im, input_point, input_label, input_box, otsu_centroid, clase, punto, cfg)
    except torch.cuda.OutOfMemoryError:
        print(f"[{clase}/{nombre}] Sin memoria GPU (crop {w0}x{h0}, enviado como {w}x{h}); se omite")
        torch.cuda.empty_cache()
        continue

    sospecha = motivos_sospecha(mask, clase)
    if sospecha and REINTENTAR_CON_CROP_ORIGINAL and os.path.exists(crop_path):
        crop_orig = cv2.imread(crop_path)
        if crop_orig is not None:
            if crop_orig.shape[:2] != (h, w):
                crop_orig = cv2.resize(crop_orig, (w, h), interpolation=cv2.INTER_AREA)
            try:
                mask2, score2, motivo2, area2 = segmentar(
                    crop_orig, input_point, input_label, input_box, otsu_centroid, clase, punto, cfg)
                sospecha2 = motivos_sospecha(mask2, clase)
                if not sospecha2:
                    print(f"[{clase}/{nombre}] Máscara sospechosa ({'; '.join(sospecha)}); "
                          f"el reintento sobre el crop original dio una máscara aceptable")
                    mask, mask_score, motivo, area_antes = mask2, score2, motivo2 + " [reintento con crop original]", area2
                    sospecha = []
                else:
                    print(f"[{clase}/{nombre}] El reintento sobre el crop original tampoco dio una máscara aceptable")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

    estado = "OK" if not sospecha else "REVISAR: " + "; ".join(sospecha)
    print(f"[{clase}/{nombre}] score={mask_score:.3f}, área={area_antes:.1%} -> {mask.sum() / mask.size:.1%} "
          f"tras limpieza | {tipo_caja}, {n_esquinas} esquinas neg. | {motivo} | {estado}")

    if esc < 1.0:  
        mask = cv2.resize(mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST) > 0

    final_viz = cv2.imread(crop_path) if os.path.exists(crop_path) else cv2.resize(im, (w0, h0))
    final_viz[mask] = final_viz[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

    excluir = bool(sospecha) and EXCLUIR_SOSPECHOSAS
    if sospecha:
        revisar.append(f"{clase}/{nombre}\t{'; '.join(sospecha)}")

    out_dir = os.path.join(revisar_dir if excluir else output_root, clase)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"resultado_final_{nombre}.jpg"), final_viz)

    if not excluir:
        os.makedirs(os.path.join(masks_dir, clase), exist_ok=True)
        cv2.imwrite(os.path.join(masks_dir, clase, f"{nombre}.png"), mask.astype(np.uint8) * 255)

if revisar:
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "_revisar.txt"), "w") as f:
        f.write("\n".join(revisar))

completas_dir = os.path.join(output_root, "completas")
os.makedirs(completas_dir, exist_ok=True)

clases = sorted(d for d in os.listdir(crops_root) if os.path.isdir(os.path.join(crops_root, d)))
paleta = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]  
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
print(f"Segmentacion lista ({len(preprocesado_paths)} imagenes procesadas, {len(revisar)} marcadas para revisar)")
