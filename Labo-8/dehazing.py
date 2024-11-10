from scipy import ndimage
from skimage import img_as_ubyte
from skimage import img_as_float
from skimage import util
from skimage.color import rgb2hsv

import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os
import PIL.Image as pill

#imagenFilename = os.path.join("./Imagenes_de_pruebas/standard_test_images/", "niebla0.png")
imagenPruebaColorA = pill.open("./Imagenes_de_pruebas/standard_test_images/niebla0.png")
imagenPruebaColor = np.asarray(imagenPruebaColorA.convert('RGB'))

def distancias(imagen):
    imagenhsv = rgb2hsv(imagen)
    r = 15
    tita0 = 0.121779
    tita1 = 0.0959710
    tita2 = -0.780245
    sigma = 0.041337

    canalv = imagenhsv[:,:,2]
    canals = imagenhsv[:,:,1]
    epsilon = np.random.normal(0,sigma,imagen.shape)

    distancias = tita0 + tita1*canalv + tita2*canals + epsilon

    distanciasFiltradas = ndimage.minimum_filter(distancias,size=r)

    return distanciasFiltradas

def dehaze(imagen):
    pass

distanciaImagenPruebaColor = distancias(imagenPruebaColor)

axs, fig = plt.subplots(1,3, figsize=(10,10))
axs[0].imshow(imagenPruebaColor)
axs[1].imshow(distanciaImagenPruebaColor)