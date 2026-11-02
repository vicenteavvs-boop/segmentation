import cv2
import numpy as np
import glob
import os
import json
import shutil
 
crops_root = '/home/fvalencia/nreyes/galaxy/runs/detect/predict/crops'
output_root = '/home/fvalencia/nreyes/galaxy/preprocesado'
 
MIN_AREA_RATIO_CONFIABLE = 0.01  
 
if os.path.exists(output_root):
    shutil.rmtree(output_root)
 
image_paths = sorted(glob.glob(os.path.join(crops_root, '*', '*.jpg')))
print(f"Se encontraron {len(image_paths)} imagenes en {len(set(os.path.dirname(p) for p in image_paths))} clases")
 
for path in image_paths:
    clase = os.path.basename(os.path.dirname(path))
    nombre = os.path.splitext(os.path.basename(path))[0]
 
    im = cv2.imread(path)
    if im is None:
        print(f"Aviso: no se pudo leer {path}, se omite")
        continue
 
    h, w = im.shape[:2]
 
    # filtro gris y blur
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)
 
    # otsu para binarizado
    th, im_binary_otsu = cv2.threshold(im_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(im_binary_otsu, cv2.MORPH_OPEN, kernel)
 
    # imagen input
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(im_gray)
 
    # combinacion, coincidencia de pixeles entre mascara y clahe
    im_input = cv2.bitwise_and(enhanced, enhanced, mask=clean_mask)
 
    #puntos de segmentación de la máscara otsu real

    area_ratio = float(np.count_nonzero(clean_mask)) / clean_mask.size
    otsu_confiable = area_ratio >= MIN_AREA_RATIO_CONFIABLE
 
    M = cv2.moments(clean_mask, binaryImage=True)
    if M["m00"] > 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    else:
        cx, cy = w / 2, h / 2 
 
    rng = np.random.default_rng(abs(hash(nombre)) % (2**32))  
 
    # puntos positivos: centroide + un par de puntos internos del objeto
    positive_points = [[float(cx), float(cy)]]
    core_mask = cv2.erode(clean_mask, np.ones((7, 7), np.uint8), iterations=2)
    core_ys, core_xs = np.where(core_mask > 0)
    if len(core_xs) > 0:
        idx = rng.choice(len(core_xs), size=min(2, len(core_xs)), replace=False)
        positive_points += [[float(core_xs[i]), float(core_ys[i])] for i in idx]
 
    # puntos negativos: fondo real segun Otsu 
    bg_ys, bg_xs = np.where(clean_mask == 0)
    if len(bg_xs) > 0:
        idx = rng.choice(len(bg_xs), size=min(4, len(bg_xs)), replace=False)
        negative_points = [[float(bg_xs[i]), float(bg_ys[i])] for i in idx]
    else:
        negative_points = [[w * 0.02, h * 0.02], [w * 0.98, h * 0.02],
                            [w * 0.02, h * 0.98], [w * 0.98, h * 0.98]]
 
    out_dir = os.path.join(output_root, clase)
    os.makedirs(out_dir, exist_ok=True)
 
    out_path = os.path.join(out_dir, f"input_para_sam_{nombre}.jpg")
    cv2.imwrite(out_path, im_input)
 
    sidecar = {
        "centroid": [cx, cy],
        "positive_points": positive_points,
        "negative_points": negative_points,
        "area_ratio": area_ratio,
        "otsu_confiable": otsu_confiable,
    }
    json_path = os.path.join(out_dir, f"input_para_sam_{nombre}.json")
    with open(json_path, "w") as f:
        json.dump(sidecar, f)
 
    aviso = "" if otsu_confiable else "  [AVISO: Otsu casi no encontró objeto, revisar manualmente]"
    print(f"[{clase}] Preprocesado listo: {out_path}  (área Otsu={area_ratio:.2%}){aviso}")
 
print(f"etapa 2 terminada ({len(image_paths)} imagenes procesadas)")
 
