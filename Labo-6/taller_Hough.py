import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
from skimage.draw import line 
from scipy import ndimage

def BuildImage():

    image_point2 = np.zeros((50,50))

    image_point2[20,20] = 255
    image_point2[40,40] = 255

    image_point3 = np.zeros((50,50))

    image_point3[20,20] = 255
    image_point3[45,45] = 255
    image_point3[12,35] = 255

    image_point4 = np.zeros((50,50))

    image_point4[10,20] = 255
    image_point4[40,45] = 255
    image_point4[10,30] = 255
    image_point4[30,30] = 255

    image_point5 = np.zeros((50,50))

    image_point5[25,20] = 255
    image_point5[45,45] = 255
    image_point5[1,20] = 255
    image_point5[48,20] = 255
    image_point5[34,38] = 255


    return  image_point2, image_point3, image_point4, image_point5

def trasnformdaHough(imagen):
    #Se recorren todas las posibles rectas entre phi=-sqrt(2)*50 a phi=sqrt(2)*50 y entre tita=-90° hasta tita=90°
    limphi = np.round(50*np.sqrt(2))
    limtita = 90
    titas = np.arange(-limtita,limtita,1)
    phies = np.arange(-limphi,limphi,1)
    #Calculo de acumulada
    acumulada =  np.zeros((phies.size,titas.size))
    for x in range(imagen.shape[0]):
        for y in range(imagen.shape[1]):
            if (imagen[x,y] == 255):
                for i in range(titas.size):
                    tita = titas[i]
                    #Se transforma a radianes
                    titarad = tita*np.pi/180
                    #Se calcula el valor de phi para ese angulo redondeando a la unidad
                    phi = np.round(x*np.cos(titarad) + y*np.sin(titarad))
                    #Buscamos el indice de ese valor de phi en la lista de phies que se toman en cuenta
                    phinorm = np.where(phies == phi)
                    if(phi in phies):
                        #Si el phi está, se suma uno a la acumulada
                        acumulada[phinorm,i] += 1
    #Dado que se buscan lineas entre puntos, solo nos interesan las rectas con al menos 2 votos
    acumulada_filtrada = 1<acumulada

    #Aqui buscamos representar las líneas en la imagen
    imagen_lineas = np.zeros(imagen.shape)
    for i in range(titas.size):
        for j in range(phies.size):
            phi = phies[j]
            tita = titas[i]
            titarad = tita*np.pi/180
            m = 0
            b = phi
            #Si para esos valores de phi y tita la acumulada filtrada es 1 entonces la recta es relevante
            if(acumulada_filtrada[j,i] == 1):
                #Separamos los casos tita=0 (cuando la recta es de la forma x=cte) de los casos tita!=0 (se trata con las formulas normalmente)
                if(titarad != 0):
                    m = - np.cos(titarad)/np.sin(titarad)
                    b = phi/np.sin(titarad)
                    #Para cada valor de x en la imagen se obtiene su respectivo valor de y
                    for x in range(imagen.shape[0]):
                        y = int(np.round(m*x + b))
                        #Varificamos que el valor de y calculado esté en rango
                        if(0<=y<imagen_lineas.shape[1]):
                            #Se saturan los puntos en la imagen correspondientes a la recta
                            imagen_lineas[x,y] = 255
                else:
                    #En el caso tita = 0, se saturan los pixeles correspondientes a x=phi
                    x = int(np.round(phi))
                    if(0<=x<imagen.shape[0]):
                        for y in range(imagen.shape[1]):
                            imagen_lineas[x,y] = 255

    return imagen_lineas, acumulada

def plot():
    #Imagenes con 2, 3, 4, 5 puntos respectivamente
    imagepoint2, imagepoint3, imagepoint4, imagepoint5 = BuildImage()
    Houghpoint2, acumpoint2 = trasnformdaHough(imagepoint2)
    Houghpoint3, acumpoint3 = trasnformdaHough(imagepoint3)
    Houghpoint4, acumpoint4 = trasnformdaHough(imagepoint4)
    Houghpoint5, acumpoint5 = trasnformdaHough(imagepoint5)
    fig, axes = plt.subplots(3,4, figsize = (10,10))
    axes = axes.ravel()
    axes[0].imshow(imagepoint2, cmap=plt.cm.gray)
    axes[1].imshow(imagepoint3, cmap=plt.cm.gray)
    axes[2].imshow(imagepoint4, cmap=plt.cm.gray)
    axes[3].imshow(imagepoint5, cmap=plt.cm.gray)


    axes[4].imshow(Houghpoint2, cmap=plt.cm.gray)
    axes[5].imshow(Houghpoint3, cmap=plt.cm.gray)
    axes[6].imshow(Houghpoint4, cmap=plt.cm.gray)
    axes[7].imshow(Houghpoint5, cmap=plt.cm.gray)

    axes[8].imshow(acumpoint2, cmap=plt.cm.gray)
    axes[9].imshow(acumpoint3, cmap=plt.cm.gray)
    axes[10].imshow(acumpoint4, cmap=plt.cm.gray)
    axes[11].imshow(acumpoint5, cmap=plt.cm.gray)


    plt.show()
    return 0
plot()

