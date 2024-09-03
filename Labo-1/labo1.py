import cv2
import numpy as np
import matplotlib.pyplot as plt

path_output = "./salida/" 


imagen =cv2.imread("Imagenes_para_contraste/galaxy.jpg", cv2.IMREAD_COLOR)

# dimension de la imagen
#print(np.shape(imagen))


def multiplicar(imagen, factor):
    # normalizamos la imagen
    imagen2 = np.float32(imagen)/256
    # multiplicacmos por el factor deseado y clipeamos el valor para que no se pase del rango de colores
    # finalmente desnormalizamos
    cv2.imwrite(path_output+"mult"+str(factor)+".jpg", np.clip(imagen2*factor,0,1)*256)

def negativo(imagen):
    imagen2 = 255-imagen
    cv2.imwrite(path_output + "negativo.jpg", imagen2)

def histograma(imagen):
    bucket = np.zeros(256)
    for line in imagen:
        for pixel in line:
            bucket[pixel] += 1
    plt.bar(np.arange(256), bucket, width=1.0)
    #plt.hist(imagen.flatten())
    plt.show()

# devuelve la función de la suma frecuencia acumulada de la imagen pasada por parametro
def funAcumulada(imagen):
    bucket = np.zeros(256)
    for line in imagen:
        for pixel in line:
            bucket[pixel] += 1
    
    frecuencias = bucket/(imagen.shape[0]*imagen.shape[1])
    funcAcumulada = np.cumsum(frecuencias)
    return funcAcumulada


def contraste(imagen):
    funcAcumulada = funAcumulada(imagen)
    funcAcumUniforme = np.arange(256)/255 #Empieza en 0????????
    transformacion = np.zeros(256)

    # calculamos la transformacion para "uniformizar" la acumulada de la distribución de la imagen original
    # (es una implementación de lo visto en clase)
    for i in range(funcAcumulada.size):
        w = funcAcumulada[i]
        diffmin = np.inf
        indicemin = 0
        for j in range(funcAcumUniforme.size):
            wn = funcAcumUniforme[j]
            diff = wn - w
            if(diff>=0 and diff<diffmin):
                diffmin = diff
                indicemin = j
        transformacion[i] = indicemin

    # aplicamos la transformacion a los pixeles de la imagen original
    imagen2 = imagen
    for i in range(imagen2.shape[0]):
        for j in range(imagen2.shape[1]):
            imagen2[i][j] = transformacion[imagen2[i][j]]
    
    cv2.imwrite(path_output + "contrasta2.jpg", imagen2)
    return imagen2
    # para ver el histograma de la resultante de la transformacion
    #plt.plot(funAcumulada(imagen2))
    #plt.show()



def binarizador(img , umbral):
    imagen2 = img

    # hago una mascara de los pixeles que estan por arriba del umbra
    # los que estan por encima le asigno el nivel de gris 255 y los que no los dejo en 0 
    imagen2 = ((np.ones(imagen.shape)*umbral)<=img)  *255
    cv2.imwrite(path_output + "binariza2.jpg", imagen2)
   # histograma(imagen2)
   # histograma(img)

# ejercicio 7
def comparar(img):
    histograma(img)
    histograma(contraste(img))





def tests():
    multiplicar(imagen,2)
    multiplicar(imagen,3)
    multiplicar(imagen,4)
    multiplicar(imagen,255)
    
    negativo(imagen)

    histograma(imagen)

    contraste(imagen)
    binarizador(imagen,102)

    