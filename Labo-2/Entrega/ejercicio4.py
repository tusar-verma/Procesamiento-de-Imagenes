import ejercicio3
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

path_output = "./salida/" 

imagen =cv2.imread("./cameraman.jpg", cv2.IMREAD_GRAYSCALE)



def filtro_media(imagen, tam_filtro):
    vector_filtro = np.ones(tam_filtro).reshape(tam_filtro, 1) 
    
    filtro = (vector_filtro @ vector_filtro.T) / (tam_filtro * tam_filtro)
    
    res = ejercicio3.conv_discreta(imagen, filtro)
    
    cv2.imwrite(path_output + "media-" + str(tam_filtro) + ".jpg", res)
    
def filtro_media_ej_c(imagen):
    filtro = np.array([[1,1],[1,1],[1,1]])
    res = ejercicio3.conv_discreta(imagen, filtro)
    
    cv2.imwrite(path_output + "media-ej4c" + ".jpg", res)
    
def func_gauss_separada(x, sigma):
    return math.e**-(x**2 / 2*(sigma**2))

def filtro_gauss(imagen, tam, sigma):
    filtro_x = np.ones(tam)
    filtro_y = np.ones(tam)
    
    for i in range(-(tam//2), tam//2 +1):
        filtro_x[i+(tam//2)] = func_gauss_separada(i, sigma)
        filtro_y[i+(tam//2)] = filtro_x[i+(tam//2)]
    
    filtro_x = filtro_x.reshape(filtro_x.shape[0], 1)
    filtro_y = filtro_y.reshape(1, filtro_y.shape[0])
    mat_filtro = filtro_x @ filtro_y
    
    factor_normalizacion = np.sum(mat_filtro)
    
    ### matriz de filtro
    #mat_filtro = mat_filtro / factor_normalizacion
    #res = ejercicio3.conv_discreta(imagen, mat_filtro)
   
   
    ### vectores de filtro (por separado)
    res = ejercicio3.conv_discreta(imagen, filtro_x)
    res = ejercicio3.conv_discreta(res, filtro_y)
    res = res / factor_normalizacion
    
    cv2.imwrite(path_output + "gauss_tam" + str(tam) + "_sigma_" + str(sigma) + "_separado" + ".jpg", res)
    
    
# tipo 
# 0: min
# 1: max
# 2: mediana

def filtro_mmm(imagen, tam, tipo):    
    
    if tipo == 0:
        operacion = np.min
        str_salida = "minimo_"
    elif tipo == 1:
        operacion = np.max
        str_salida = "maximo_"
    else:
        operacion = np.median
        str_salida = "mediana_"
    
    imagen_con_padding = np.pad(imagen, [(0, tam-1),(0, tam- 1)])
    
    m, n = imagen.shape
    
    res = np.zeros(imagen.shape)
    
    for i in range(m):
        for j in range(n):
            ventana = imagen_con_padding[i:i+tam, j:j+tam]            
            res[i, j] = operacion(ventana)
            
    cv2.imwrite(path_output + str_salida + "tam_" + str(tam) + ".jpg", res)

    
def test():
    #filtro_media(imagen, 3)
    #filtro_media(imagen, 21)
    #filtro_media_ej_c(imagen)


    #filtro_gauss(imagen, 3, 1)
    #filtro_gauss(imagen, 9, 1)


    #filtro_mmm(imagen, 3, 0)
    #filtro_mmm(imagen, 9, 0)

    #filtro_mmm(imagen, 3, 1)
    #filtro_mmm(imagen, 9, 1)

    #filtro_mmm(imagen, 3, 2)
    #filtro_mmm(imagen, 9, 2)
    return