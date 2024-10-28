import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
from skimage.draw import line 


def BuildImage():
    image_line = np.zeros((50,50))
    image_point = np.zeros((50,50))

    rr,cc = line(20,20,40,40)

    image_line[rr,cc] = 255

    image_point[20,20] = 255

    image_point[40,40] = 255

    image_point[20,20] = 255

    image_point[30,30] = 255

    return image_line , image_point

def trasnformdaHouf(imagen, limtita=90, limphi=50):

    #calculo de acumulada
    acumulada =  np.zeros((limphi,limtita))

    for x in range(imagen.shape[1]):
        for y in range(imagen.shape[0]):
            if (imagen[y,x] == 255):
                for tita in range(limtita):
                    phi = x*np.cos(tita) + y*np.sin(tita)
                    if(phi<limphi):
                        acumulada[phi,tita] += 1
    
    acumulada_filtrada = 255 * np.ones(acumulada.shape) * (acumulada > np.max(acumulada)*0.75)
    
    #representar las lineas en la imagen
    imagen_lineas = np.zeros(imagen.shape)
    for tita in range(limtita):
        for phi in range(limphi):
            m = - np.cos(tita)/np.sin(tita)
            b = phi/np.sin(tita)
            if(acumulada_filtrada[phi,tita] == 255):
                for x in range(imagen.shape[1]):
                    y = m*x + b
                    if(y<imagen_lineas.shape[0]):
                        imagen_lineas[y,x] = 255


    return imagen_lineas

def plot():
    fig, axes = plt.subplot(1,2)
    return
