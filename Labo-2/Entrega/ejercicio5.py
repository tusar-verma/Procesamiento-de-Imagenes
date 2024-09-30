import ejercicio3
import ejercicio4
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


path_output = "./salida/unsharp_masking/" 

imagen =cv2.imread("./cameraman.jpg", cv2.IMREAD_GRAYSCALE)

def unsharp_mask(imagen, a, sigma):
    imagen_padding = np.pad(imagen, [(0, 2),(0, 2)])
    gauss_img = ejercicio4.filtro_gauss(imagen, 3, sigma)
    
    mask = imagen_padding - gauss_img
    
    res = imagen_padding + a * mask
    
    cv2.imwrite(path_output + "unsharp_masking_sigma_" + str(sigma) + "_factor_" + str(a) + ".jpg", res)
    
def test():    
    unsharp_mask(imagen, 1, 1)
    unsharp_mask(imagen, 1, 2.5)
    unsharp_mask(imagen, 1, 10)
    unsharp_mask(imagen, 1, 20)
    unsharp_mask(imagen, 2.5, 10)
    unsharp_mask(imagen, 2.5, 15)
    unsharp_mask(imagen, 5, 2.5)
    unsharp_mask(imagen, 10, 5)
    unsharp_mask(imagen, 10, 10)
    
test()