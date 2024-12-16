import numpy as np
import os
import cv2 as cv
from skimage import feature
from skimage import img_as_float, img_as_ubyte
import mosaic as mo
import random
import matplotlib.pyplot as plt


def mati_dice(i1,i2):
    m1 = i1.shape[0]
    n1 = i1.shape[1]
    m2 = i2.shape[0]
    n2 = i2.shape[1]

    M = max(m1,m2)
    N = max(n1,n2)
    
    im1padd = -np.ones((M,N,3))
    im2padd = -2*np.ones((M,N,3))

    im1padd[:m1,:n1,:] = i1
    im2padd[:m2,:n2,:] = i2
    
    bm =((im1padd[:,:,0] == im2padd[:,:,0]) & (im1padd[:,:,1] == im2padd[:,:,1]) & (im1padd[:,:,2] == im2padd[:,:,2]))
    dice_coeff = np.sum(bm)/(im1padd.shape[0]*im2padd.shape[1])
    return dice_coeff

def dice_promedio():
    i = 0
    coef_dice = 0
    for imagen in os.listdir("imagenes/Forest"):
        
        #leeo la imagen
        if(i < 61 ): #1999*0.03
            imagen_original = cv.imread(f"imagenes/Forest/{imagen}", cv.IMREAD_COLOR)
            mitad = int(imagen_original.shape[1] * (1/2)) +17
            porcion = random.randint(mitad, imagen_original.shape[1])

            #dos_tercios = int(imagen_original.shape[1] * porcion)
            dos_tercios = porcion
            un_tercio   = imagen_original.shape[1] - dos_tercios
            print(dos_tercios, un_tercio)
            imagen_1 = imagen_original[0:imagen_original.shape[0],0:dos_tercios+1]
            imagen_2 = imagen_original[0:imagen_original.shape[0],un_tercio:imagen_original.shape[1]]
            print(imagen_1.shape,imagen_2.shape,imagen_original.shape)
            imagen_generada = mo.mosaico(imagen_1,imagen_2)
            imagen_generada = img_as_ubyte(imagen_generada)
            cv.imwrite(f"resultados_dice/{i}.png",imagen_generada)
            
            dice = mati_dice(imagen_generada,imagen_original)
            print(dice)
            coef_dice += dice
            i+=1
            

    coef_dice = coef_dice / i
    return coef_dice


def mati_thresh(i1,i2):
    m1 = i1.shape[0]
    n1 = i1.shape[1]
    m2 = i2.shape[0]
    n2 = i2.shape[1]
    thresh = 15
    M = max(m1,m2)
    N = max(n1,n2)
    
    im1padd = -(1+thresh)*np.ones((M,N,3))
    im2padd = -(1+2*thresh)*np.ones((M,N,3))

    im1padd[:m1,:n1,:] = i1
    im2padd[:m2,:n2,:] = i2
    
    bm =((np.abs(im1padd[:,:,0] - im2padd[:,:,0] )<thresh)& (np.abs(im1padd[:,:,1] -im2padd[:,:,1])<thresh) & (np.abs(im1padd[:,:,2] - im2padd[:,:,2]) < thresh))
    dice_coeff = np.sum(bm)/(im1padd.shape[0]*im2padd.shape[1])
    return dice_coeff

def thresholding(radio,exigencia,cc):
    i = 0
    coef_dice = 0
    lista_coef = open(f"resultados_{radio}_{exigencia}.txt","w")
    for imagen in os.listdir("imagenes/Forest"):
        
        #leeo la imagen
        if(i < 30 ): #1999*0.03
            lista_coef.write(f"{imagen}:")
            imagen_original = cv.imread(f"imagenes/Forest/{imagen}", cv.IMREAD_COLOR)
            mitad = int(imagen_original.shape[1] * (1/2)) +17
            porcion = random.randint(mitad, imagen_original.shape[1])

            #dos_tercios = int(imagen_original.shape[1] * porcion)
            dos_tercios = int(imagen_original.shape[1] * (2/3))
            un_tercio   = imagen_original.shape[1] - dos_tercios
            
            imagen_1 = imagen_original[0:imagen_original.shape[0],0:dos_tercios+1]
            imagen_2 = imagen_original[0:imagen_original.shape[0],un_tercio:imagen_original.shape[1]]
            
            imagen_generada = mo.mosaico(imagen_1,imagen_2,radio,exigencia,cc)
            imagen_generada = img_as_ubyte(imagen_generada)
            cv.imwrite(f"resultados_dice/{i}.png",imagen_generada)
            
            dice = mati_thresh(imagen_generada,imagen_original)
            lista_coef.write(f"{dice}\n")
            
            coef_dice += dice
            i+=1
            

    coef_dice = coef_dice / i
    lista_coef.write(f"promedio:{coef_dice}")
    return coef_dice

list_ventana = [1,3,5,10]
list_exigencia = [0.7,0.8,0.9]
list_coefCorr = [0.5, 0.7, 0.8, 0.9]
matri_res = np.zeros((len(list_ventana),len(list_exigencia), len(list_coefCorr ))) 
res_mat = open(f"resultados_matri_res.txt","w")

for i in range(len(list_ventana)):
    for j in range(len(list_exigencia)):
        for k in range(len(list_coefCorr)):
            matri_res[i,j,k] = thresholding(list_ventana[i],list_exigencia[j], list_coefCorr[k])
            res_mat.write(str(matri_res[i,j,k])+" ")
        res_mat.write("\n")
    res_mat.write("\n")


plt.imshow(matri_res)
plt.colorbar()
plt.show()
