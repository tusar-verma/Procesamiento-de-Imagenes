
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

imagenPruebaColorA = pill.open("./Imagenes_de_pruebas/standard_test_images/niebla0.png")
imagenPruebaColor = img_as_float(np.asarray(imagenPruebaColorA.convert('RGB')))

#Se calculan las distancias con el método propuesto
def distancias(imagen):
    imagenhsv = rgb2hsv(imagen)
    r = 15
    tita0 = 0.121779
    tita1 = 0.959710
    tita2 = -0.780245
    sigma = 0.041337

    canalv = imagenhsv[:,:,2]
    canals = imagenhsv[:,:,1]
    epsilon = np.random.normal(0,sigma,canalv.shape)
    
    #Calculo de distancia
    distancias = tita0 + tita1*canalv + tita2*canals + epsilon # Con epsilon=0 se ve con menos ruido en las distancias

    #Filtro para evitar artefactos con colores blancos que se pueden erroneamente interpretar como distantes
    distanciasFiltradas = ndimage.minimum_filter(distancias,size=r)

    return distanciasFiltradas

def dehaze(imagen):
    beta = 1
    percentil = 0.001
    m = imagen.shape[0]
    n = imagen.shape[1]
    mapaDistancias = distancias(imagen)
    imagenhsv = rgb2hsv(imagen)
    
    #Obtenemos el A que resulta de tomar dentro del 0.1% pixeles más brillantes en el mapa de distancias, aquel pixel con mayor intensidad como A
    listaImagenv = imagenhsv[:,:,2].flatten()
    listaDistancias = mapaDistancias.flatten()
    #Obtenemos los indices de los valores en listaDistancias ordenados de mayor a menor filtrando al percentil 0.1
    indices = np.flip(np.argsort(listaDistancias))[:int(listaDistancias.size*percentil)]
    
    #Buscamos las intesidades de esos indices
    intensidades = listaImagenv[indices]

    #Obtenemos el indice correspondiente a la maxima intensidad en listaImagenv
    indmax = indices[np.argmax(intensidades)]

    #Deducimos las coordenadas a partir del indice en la "matriz planchada"
    coordenadaspixel = ((indmax//n), indmax%n)

    A = imagenhsv[:,:,2][coordenadaspixel] 

    #Reconstruimos la imagen sin niebla
    mA = np.ones((m,n))*A
    mt = np.clip(np.exp(-beta*mapaDistancias),0.1,0.9)
    J = imagen.copy()
    J[:,:,0] = (imagen[:,:,0] - mA)/mt + mA
    J[:,:,1] = (imagen[:,:,1] - mA)/mt + mA
    J[:,:,2] = (imagen[:,:,2] - mA)/mt + mA

    J = np.clip(J,0,1)
    return J

distanciaImagenPruebaColor = distancias(imagenPruebaColor)
dehazedImagenPruebaColor = dehaze(imagenPruebaColor)
fig, axs = plt.subplots(1,3, figsize=(10,10))
#Imagen original
axs[0].imshow(imagenPruebaColor)
#Mapa de distancias
axs[1].imshow(distanciaImagenPruebaColor, cmap="inferno")
#Imagen sin niebla
axs[2].imshow(dehazedImagenPruebaColor)
plt.show()