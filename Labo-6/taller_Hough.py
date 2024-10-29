import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
from skimage.draw import line 


def BuildImage():
    image_line = np.zeros((50,50))
    image_point = np.zeros((50,50))

    rr,cc = line(20,20,40,40)

    image_line[rr,cc] = 255

    rr,cc = line(10,45,40,45)

    image_line[rr,cc] = 255

    image_point[20,20] = 255

    image_point[40,40] = 255

    image_point[20,20] = 255

    image_point[30,30] = 255

    return image_line , image_point

def trasnformdaHough(imagen, liminftita=-90, limsuptita=90, limphi=50):
    titas = np.arange(liminftita,limsuptita)

    #calculo de acumulada
    acumulada =  np.zeros((limphi,titas.size))
    for x in range(imagen.shape[0]):
        for y in range(imagen.shape[1]):
            if (imagen[x,y] == 255):
                for tita in titas:
                    titarad = tita*np.pi/180
                    titanorm = tita + 90
                    phi = int(x*np.cos(titarad) + y*np.sin(titarad))
                    if(0<=phi<limphi):
                        acumulada[phi,titanorm] += 1
    
    acumulada_filtrada = 255 * np.ones(acumulada.shape) * (acumulada >= np.max(acumulada)*0.65)
    
    #representar las lineas en la imagen
    imagen_lineas = np.zeros(imagen.shape)
    for tita in titas:
        for phi in range(limphi):
            titarad = tita*np.pi/180
            titanorm = tita + 90
            m = 0
            b = phi
            if(titarad != 0):
                m = - np.cos(titarad)/np.sin(titarad)
                b = phi/np.sin(titarad)
            if(acumulada_filtrada[phi,titanorm] == 255):
                for x in range(imagen.shape[1]):
                    y = int(m*x + b)
                    if(0<=y<imagen_lineas.shape[0]):
                        imagen_lineas[x,y] = 255


    return imagen_lineas, acumulada

def plot():
    imageline, imagepoint = BuildImage()
    Houghline, acumline = trasnformdaHough(imageline)
    Houghpoint, acumpoint = trasnformdaHough(imagepoint)
    fig, axes = plt.subplots(3,2, figsize = (10,10))
    axes = axes.ravel()
    axes[0].imshow(imageline, cmap=plt.cm.gray)
    axes[1].imshow(imagepoint, cmap=plt.cm.gray)
    axes[2].imshow(Houghline, cmap=plt.cm.gray)
    axes[3].imshow(Houghpoint, cmap=plt.cm.gray)
    axes[4].imshow(acumline, cmap=plt.cm.gray)
    axes[5].imshow(acumpoint, cmap=plt.cm.gray)

    plt.show()
    return 0
plot()
