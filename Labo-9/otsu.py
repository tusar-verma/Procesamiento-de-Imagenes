import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

rice_image = cv.imread("Imagenes/rice.png", cv.IMREAD_GRAYSCALE)
path_res = "./resultados_otsu/"
def Otsu(imagen):
    #calculo el normalized histogram
    histograma = cv.calcHist([imagen],[0],None,[256],[0,256])
    histograma = histograma / (imagen.shape[0] * imagen.shape[1])
    
    # cumlative sums
    
    # probabilidad de la clase 1
    P_1 = np.zeros(histograma.shape)

    P_1[0] = histograma[0]

    for i in range(1 , histograma.shape[0]):
        P_1[i] = P_1[i-1] + histograma[i]
        
    # En M se guardan las medias acumuladas
    M = np.zeros(histograma.shape[0])

    M[0] = 0

    for i in range(1,histograma.shape[0]):
        M[i] = (M[i-1] + (i * histograma[i]))[0]
    
    # Media global
    M_g = M[histograma.shape[0] - 1]

    # between-class variance
    var_for_k = np.zeros(histograma.shape[0], dtype=int)

    for i in range(histograma.shape[0]):
        
        var = 0
        #Se calcula varianza por cada valor de intensidad
        if(P_1[i] != 0 and P_1[i] < 0.999999999999999):
    
            var = ((np.power((M_g * P_1[i]) - M[i],2)) / (P_1[i]*(1-P_1[i])))[0]
        var_for_k[i] = var

    # optimo umbral k
    optimo_k = np.argmax(var_for_k)
    
    #Se segmenta la imagen, según el threshld óptimo
    imagen_segmentada = imagen.copy()

    for x in range(imagen_segmentada.shape[0]):
        for y in range(imagen_segmentada.shape[1]):   
            if(imagen[x , y] > (optimo_k ) ):
                imagen_segmentada[x , y] = 255
            else:
                imagen_segmentada[x , y] = 0

    return imagen_segmentada,optimo_k

ret,th1 = cv.threshold(rice_image,127,255,cv.THRESH_BINARY)

#arroz segmentado de OpenCV
cv.imwrite(path_res + "arroz_segmentado_de_cv2.png",th1)
resImage,k = Otsu(rice_image)
#Arroz segmentado propio
cv.imwrite(path_res + "arroz_segmentado.png",resImage)

