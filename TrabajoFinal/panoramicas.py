import numpy as np
import os
import cv2 as cv
from skimage import feature
from skimage import img_as_float, img_as_ubyte
import mosaic as mo
import random

def dana():
    i = 0
    img_pan = os.listdir("imagenes/dana")
    img_pan.sort()
    for pan in img_pan:
        lista_partes = os.listdir(f"imagenes/dana/{pan}")
        lista_partes.sort()
        imagen_generada = 0 
        for i in range(len(lista_partes)):
            if(i==0):
                imagen_generada = cv.imread(f"imagenes/dana/{pan}/{lista_partes[i]}", cv.IMREAD_COLOR)
                continue
            imagen_2 = cv.imread(f"imagenes/dana/{pan}/{lista_partes[i]}", cv.IMREAD_COLOR)
            imagen_generada = mo.mosaico(imagen_2,imagen_generada)
        
        imagen_generada = img_as_ubyte(imagen_generada)
        cv.imwrite(f"resultados_dana/{pan}.png",imagen_generada)
dana()