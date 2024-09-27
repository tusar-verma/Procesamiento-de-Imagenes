import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path_output = "salida/" 


imagen =cv2.imread("./Imagenes_para_contraste/vb.jpg", cv2.IMREAD_GRAYSCALE)

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

    return bucket

def graficar_histograma(imagen):
    bucket = histograma(imagen)
    plt.bar(np.arange(256), bucket, width=1.0)
    #plt.hist(imagen.flatten())
    plt.show()
    

# devuelve la función de la suma frecuencia acumulada de la imagen pasada por parametro
def funAcumulada(imagen):
    bucket = np.zeros(256)
    for line in imagen:
        for pixel in line:
            bucket[pixel] += 1
    
    funcAcumulada = np.cumsum(bucket)/(imagen.shape[0]*imagen.shape[1])
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
    imagen2 = imagen.copy()
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

    # hago una mascara de los pixeles que estan por arriba del umbral
    # a los que estan por encima les asigno el nivel de gris 255 y a los que no los dejo en 0 
    imagen2 = ((np.ones(imagen.shape)*umbral)<=img)  *255
    cv2.imwrite(path_output + "binariza2.jpg", imagen2)
   # graficar_histograma(imagen2)
   # graficar_histograma(img)

# ejercicio 7
def comparar(img):
    graficar_histograma(img)
    graficar_histograma(contraste(img))

# ejercicio 8
def ecualizarDosVeces(img):
    newImg = contraste(img)
    return contraste(newImg)

# ejercicio 9
def modhistograma(lanmda, gamma, img):
    # calculo el histograma de la imagen
    h0 = histograma(img)
    # lo transformo en un vector columna
    h0 = h0.reshape((h0.size,1))
    # histograma uniforme antes de normalizar
    u = np.ones(256) * (np.shape(img)[0]*np.shape(img)[1]) 
    u = u.reshape((u.size,1))

    # hago la matriz bidiagona
    bidiag = np.eye(255, 256) * -1 +  np.diag(np.ones(255) , 1)[:255,:]
    # minimizacion (diapo)
    res = np.linalg.solve(256*(np.eye(np.shape(bidiag)[1],np.shape(bidiag)[1]) * (1+lanmda) + gamma*bidiag.T@bidiag), (256*h0 + (lanmda*u)))

    return res

def transformar_imagen(imagen, funcAcumTarget):
    funcAcumulada = funAcumulada(imagen)
    transformacion = np.zeros(256)

    for i in range(funcAcumulada.size):
        w = funcAcumulada[i]
        diffmin = 3
        indicemin = 255
        for j in range(funcAcumTarget.size):
            wn = funcAcumTarget[j]
            diff = wn - w
            if(i==255):
                print(wn, w, diff)
            if(diff>=0 and diff<diffmin):
                diffmin = diff
                indicemin = j
               
        transformacion[i] = indicemin
        print(indicemin)

    imagen2 = imagen.copy()
    for i in range(imagen2.shape[0]):
        for j in range(imagen2.shape[1]):
            imagen2[i][j] = transformacion[imagen2[i][j]]

    return imagen2

def exploracion9(imagen):
    lambdas = [3]
    gammas = [0]
    special_path_output = path_output + "amigo-maxi/"

    for l in lambdas:
        for g in gammas:
            h = modhistograma(l,g, imagen)
            fac = np.cumsum(h)/(imagen.shape[0]*imagen.shape[1])
            plt.plot(fac, label="lambda: " + str(l) + "\n gamma: " + str(g))
            imagen_nueva = transformar_imagen(imagen, fac)
            cv2.imwrite(special_path_output + "lambda_" + str(l) + "_gamma_" + str(g) + ".jpg" , imagen_nueva )
    plt.legend()
    plt.show()


def tests():
   # multiplicar(imagen,2)
   # multiplicar(imagen,3)
   # multiplicar(imagen,4)
   # multiplicar(imagen,255)
    
   # negativo(imagen)

   # graficar_histograma(imagen)

    contraste(imagen)
   # binarizador(imagen,102)
    
tests()
#exploracion9(imagen)