import numpy as np
import cv2 as cv

imagen = cv.imread("prueba.jpeg")

imagen_girs = cv.cvtColor(imagen,cv.COLOR_BGR2GRAY)
#imagen_girs = np.float32(imagen_girs)

harris = cv.cornerHarris(imagen_girs,10,3,0.04)
harris = harris>(0.01*np.max(harris))
esquinas = np.array([[0,0]])

for x in range(imagen.shape[0]):
    for y in range(imagen.shape[1]):
        if(harris[x,y]):
            esquina = np.array([[x,y]])

            esquinas = np.append(esquinas,[[x,y]],0)

esquinas = np.delete(esquinas,0,0)
print(esquinas[0])
