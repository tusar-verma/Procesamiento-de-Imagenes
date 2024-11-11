import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os
import PIL.Image as pill
from scipy import ndimage

def grandiente(imagen):
    filtrodx = np.array([1,1,1],[0,0,0],[-1,-1,-1])
    dx = ndimage.convolve(imagen,filtrodx)
    filtrody = np.array([-1,0,1],[-1,0,1],[-1,0,1])
    dy = ndimage.convolve(imagen,filtrody)

"""
Para la imagen primero Canny para bordes
Se OBTIENEN DERIVADAS USUALES
Luego para cada pixel borde se elige una caja
Ix es la suma se los dx de la caja punderados con una matriz gaussiana del tamaño de la caja
"""