import cv2
import numpy as np

path = ('/home/fvalencia/nreyes/galaxy/runs/detect/predict3/crops/globular_cluster/m13-s_jpg.jpg')
im = cv2.imread(path)

#filtro gris y blur
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)

#otsu para binarizado
th, im_binary_otsu  = cv2.threshold(im_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((5,5), np.uint8)
clean_mask = cv2.morphologyEx(im_binary_otsu, cv2.MORPH_OPEN, kernel)

#imagen input
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
enhanced = clahe.apply(im_gray)

#combinacion, coincidencia de pixeles entre mascara y clahe
im_input = cv2.bitwise_and(enhanced, enhanced, mask=clean_mask)

cv2.imwrite('input_para_sam.jpg', im_input)
print(f"etapa 2 terminada")

