import numpy as np
import cv2 as cv
import mosaic as m
import harris as h
from skimage import feature

imagen_der = cv.imread("der.png")
imagen_izq = cv.imread("izq.png")
imagen     = cv.imread("prueba.jpeg")

imagen_gris_der = cv.cvtColor(imagen_der,cv.COLOR_BGR2GRAY)
imagen_gris_izq = cv.cvtColor(imagen_izq,cv.COLOR_BGR2GRAY)
imagen_gris     = cv.cvtColor(imagen,cv.COLOR_BGR2GRAY)

bordes_1 = feature.canny(imagen_gris_der, sigma=3)
bordes_2 = feature.canny(imagen_gris_izq, sigma=3)
bordes_3 = feature.canny(imagen_gris, sigma=3)

puntosDeInteres_1 = list(zip(*np.where(bordes_1 == 1)))
puntosDeInteres_2 = list(zip(*np.where(bordes_2 == 1)))
puntosDeInteres_3 = list(zip(*np.where(bordes_3 == 1)))

esquinas_der = h.Harrys(imagen_gris_der,puntosDeInteres_1,0.001)
esquinas_izq = h.Harrys(imagen_gris_izq,puntosDeInteres_2,0.001)
esquinas     = h.Harrys(imagen_gris,puntosDeInteres_3,0.001)
print(esquinas)
esquinas_filtradas = m.correlacion(esquinas_der,esquinas_izq,imagen_gris_der,imagen_gris_izq)


for i in range(esquinas_filtradas.shape[0]):
    imagen_der[esquinas_filtradas[i,0,0],esquinas_filtradas[i,0,1]] = [0,255,0]
    imagen_izq[esquinas_filtradas[i,1,0],esquinas_filtradas[i,1,1]] = [0,255,0]

for i in range(esquinas.shape[0]):
    imagen[esquinas[i,0],esquinas[i,1]] = [0,255,0]

cv.imwrite("esq_der.png",imagen_der)
cv.imwrite("esq_izq.png",imagen_izq)
cv.imwrite("esquinas.png",imagen)