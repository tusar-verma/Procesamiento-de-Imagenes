import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
from skimage.draw import line 
from scipy import ndimage

def BuildImage():
    image_line = np.zeros((50,50))

    rr,cc = line(20,20,40,40)

    image_line[rr,cc] = 255

    rr,cc = line(10,45,40,45)

    image_line[rr,cc] = 255

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

    image_point5[15,20] = 255
    image_point5[45,45] = 255
    image_point5[1,20] = 255
    image_point5[12,20] = 255


    return image_line , image_point2, image_point3, image_point4, image_point5

def filtroMaxLocal(A, tam=3):
    M1 = A.shape[0]
    N1 = A.shape[1]
    diff = tam//2
    li = np.arange(-diff,diff+1)
    lj = np.arange(-diff,diff+1)
    C = np.zeros((M1, N1))
    maximoTotal = np.max(A)
    for u in range(C.shape[0]):
        for v in range(C.shape[1]):
            valores = []
            for i in li:
                for j in lj:
                    ai = u - i
                    aj = v - j
                    
                    if(0<=ai<M1 and 0<=aj<N1):
                        #if(u==0 and v==44):
                        #   print(ai,aj,A[ai,aj],i!=0,j!=0, u,v)
                        if(u!=ai or v!=aj):
                            valores.append(A[ai][aj])
            valores = np.array(valores)
            C[u,v] = np.prod(A[u,v] > valores) and A[u,v]>1 
            #if(u==0 and v==44 and C[u,v]==1):
                #print(A[u,v],A[u,v+1],valores)
            
    return C


def trasnformdaHough(imagen, limsuptita=90, limphi=50):
    titas = np.arange(-limsuptita,limsuptita)
    phies = np.arange(-limphi,limphi)
    #calculo de acumulada
    acumulada =  np.zeros((phies.size,titas.size))
    for x in range(imagen.shape[0]):
        for y in range(imagen.shape[1]):
            if (imagen[x,y] == 255):
                for tita in titas:
                    titarad = tita*np.pi/180
                    titanorm = tita + 90
                    phi = int(x*np.cos(titarad) + y*np.sin(titarad))
                    if(phi in phies):
                        phinorm = phi + limphi
                        acumulada[phinorm,titanorm] += 1
    #maximosLocales = filtroMaxLocal(acumulada,7)
    #print(np.max(acumulada))
    acumulada_filtrada = np.max(acumulada)==acumulada
    #representar las lineas en la imagen
    imagen_lineas = np.zeros(imagen.shape)
    for tita in titas:
        for phi in phies:
            titarad = tita*np.pi/180
            titanorm = tita + limsuptita
            phinorm = phi + limphi
            m = 0
            b = phi
            if(titarad != 0):
                m = - np.cos(titarad)/np.sin(titarad)
                b = phi/np.sin(titarad)
            
            if(acumulada_filtrada[phinorm,titanorm] == 1):
                
                for x in range(imagen.shape[0]):
                    y = int(m*x + b)
                    
                    if(0<=y<imagen_lineas.shape[1]):
                        
                        imagen_lineas[x,y] = 255
                        if(m==0):
                            pass

    return imagen_lineas, acumulada

def plot():
    imageline, imagepoint2, imagepoint3, imagepoint4, imagepoint5 = BuildImage()
    Houghline, acumline = trasnformdaHough(imageline)
    Houghpoint2, acumpoint2 = trasnformdaHough(imagepoint2)
    Houghpoint3, acumpoint3 = trasnformdaHough(imagepoint3)
    Houghpoint4, acumpoint4 = trasnformdaHough(imagepoint4)
    Houghpoint5, acumpoint5 = trasnformdaHough(imagepoint5)
    fig, axes = plt.subplots(3,5, figsize = (10,10))
    axes = axes.ravel()
    axes[0].imshow(imageline, cmap=plt.cm.gray)
    axes[1].imshow(imagepoint2, cmap=plt.cm.gray)
    axes[2].imshow(imagepoint3, cmap=plt.cm.gray)
    axes[3].imshow(imagepoint4, cmap=plt.cm.gray)
    axes[4].imshow(imagepoint5, cmap=plt.cm.gray)
    
    print(Houghpoint4[35,10])

    axes[5].imshow(Houghline, cmap=plt.cm.gray)
    axes[6].imshow(Houghpoint2, cmap=plt.cm.gray)
    axes[7].imshow(Houghpoint3, cmap=plt.cm.gray)
    axes[8].imshow(Houghpoint4, cmap=plt.cm.gray)
    axes[9].imshow(Houghpoint5, cmap=plt.cm.gray)

    axes[10].imshow(acumline, cmap=plt.cm.gray)
    axes[11].imshow(acumpoint2, cmap=plt.cm.gray)
    axes[12].imshow(acumpoint3, cmap=plt.cm.gray)
    axes[13].imshow(acumpoint4, cmap=plt.cm.gray)
    axes[14].imshow(acumpoint5, cmap=plt.cm.gray)


    plt.show()
    return 0
plot()

