import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


H01 = cv.imread("Imagenes/H01.bmp", cv.IMREAD_GRAYSCALE)
H02 = cv.imread("Imagenes/H02.bmp", cv.IMREAD_GRAYSCALE)
H03 = cv.imread("Imagenes/H03.bmp", cv.IMREAD_GRAYSCALE)
H04 = cv.imread("Imagenes/H04.bmp", cv.IMREAD_GRAYSCALE)
H05 = cv.imread("Imagenes/H05.bmp", cv.IMREAD_GRAYSCALE)

text_image = cv.imread("Imagenes/shaded_text.tif", cv.IMREAD_GRAYSCALE)
sine_text_image = cv.imread("Imagenes/sine_shaded_text.tif", cv.IMREAD_GRAYSCALE)


def variableTresholdingBasedOnMovingAverages(image, n=5,c=0.5):

    rows, cols = np.shape(image)
    movingAverage = np.zeros(image.shape)
    for x in range(rows):
        
        #Proceso en zig zag
        #Si la fila es impar, la invierto
        if x%2 == 0:
            rowToProcess = image[x]  
        else:
            rowToProcess = np.flip(image[x])

        pixelMA = 0
        for y in range(cols):
            
            #Contemplo los bordes para que el promedio sea sobre los pixeles disponibles
            left = max(0,y+1-n)
            right = min(cols-1,y)
            availablePixels = rowToProcess[left:right+1]

            pixelMA = np.mean(availablePixels)
            
            #Restauro el orden de las filas
            if x%2 == 0:
                movingAverage[x][y] = pixelMA 
            else:
                movingAverage[x][cols-y-1] = pixelMA
    
    threshold = movingAverage * c
    returnImage = np.zeros(image.shape,dtype=int)

    #Aplico el threshold
    for x in range(rows):
        for y in range(cols):
            if image[x,y] > threshold[x,y]:
                returnImage[x,y] = 255

    return returnImage



            
cv.imwrite("H01_segmentado.png",variableTresholdingBasedOnMovingAverages(H01,60,0.9))
cv.imwrite("H02_segmentado.png",variableTresholdingBasedOnMovingAverages(H02,60,0.9))
cv.imwrite("H03_segmentado.png",variableTresholdingBasedOnMovingAverages(H03,60,0.9))
cv.imwrite("H04_segmentado.png",variableTresholdingBasedOnMovingAverages(H04,60,0.9))
cv.imwrite("H05_segmentado.png",variableTresholdingBasedOnMovingAverages(H05,60,0.9))

cv.imwrite("shaded_text_segmentado.png",variableTresholdingBasedOnMovingAverages(text_image,20))
cv.imwrite("sine_shaded_text_segmentado.png",variableTresholdingBasedOnMovingAverages(sine_text_image,20))


