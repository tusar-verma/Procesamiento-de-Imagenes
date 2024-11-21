import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import skimage.io as io

import os

"""
En este caso entrenamos el algorítmo con imagenes de 3 deportes distintos:
Baseball, Baskbetball y Archery según se indica en el README.md de Imagenes/hog/train

Finalmente se testea el entrenamiento usando los descriptores hog con 9 imagenes de testeo:
3 por cada deporte, siguiendo el README.md de Imagenes/hog/test

En los archivos tabla.txt de las carpetas train y test estan las clasificaciones de cada grupo de imagenes 
(delimitadas por sus números) en forma de matriz.

"""

clf = svm.SVC()

path_comun = "./Imagenes/hog/"
def train():
    lista_train = []
    lista_clasif = []
    #Se obtienen la tabla con las clasificaciones
    melementos = np.loadtxt(path_comun+"train/tabla.txt", dtype=int)

    #Se construye la lista con las clasificaciones
    for fila in melementos:
        lista_clasif += [fila[2] for i in range(fila[1]-fila[0] + 1)]
    
    #Se obtienen los descriptoes de cada imagen y se añaden a la lista
    for i in range(len(lista_clasif)):
        nombre = ".jpg"
        if(i+1<10):
            nombre = "00" + str(i+1) + ".jpg"
        else:
            nombre = "0" + str(i+1) + ".jpg"
        imagenFN = os.path.join(path_comun + "train", nombre)
        imagen = io.imread(imagenFN)

        fd, hog_image = hog(
            imagen,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=True,
            channel_axis=-1,
        )

        #Se guarda descriptor
        lista_train.append(fd)
    
    #Se entrena el modelos con los descriptores hog y sus clasificaciones asociadas
    clf.fit(lista_train, lista_clasif)
def test():
    lista_clasif_real = []
    lista_a_testear = []
    #Se obtienen las clasificaciones de las imagenes de testeo
    melementos = np.loadtxt(path_comun+"test/tabla.txt", dtype=int)
    for fila in melementos:
        lista_clasif_real += [fila[2] for i in range(fila[1]-fila[0] + 1)]

    #Se obtienen los descriptores hog de las iamgenes de testeo
    for i in range(len(lista_clasif_real)):
        nombre = str(i+1) + ".jpg"
        imagenFN = os.path.join(path_comun + "test", nombre)
        imagen = io.imread(imagenFN)

        fd, hog_image = hog(
            imagen,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=True,
            channel_axis=-1,
        )
        lista_a_testear.append(fd)
    lista_clasif_real = np.array(lista_clasif_real)
    #Se obtiene la clasificación predecida por el modelo a partir de los descriptores
    lista_clasif_predict = clf.predict(lista_a_testear)
    
    #Para cada clasificación se obtienen los valores de Precision, Recall y F1-Score
    cant_tipos_distintos = melementos.shape[0]
    Precision = np.zeros(cant_tipos_distintos)
    Recall = np.zeros(cant_tipos_distintos)
    F1score = np.zeros(cant_tipos_distintos)
    for i in range(cant_tipos_distintos):
        #Nos quedamos colo con las clasificacion que nos interesa: True -> es clasificacion i, False -> No es clasificación i
        lista_clasif_real_i = lista_clasif_real == i
        lista_clasif_predict_i = lista_clasif_predict == i
        
        #Se calcula la cantidad de Positivos Verdaderos (si para imagen clasif predecida = clasif real = True)
        TruePositives = np.sum(np.logical_and(lista_clasif_predict_i, lista_clasif_real_i))
        
        #Se calcula la cantidad de Falsos Positivos (si para imagen clasif predecida = True y clasif real = False)
        FalsePositives = np.sum(np.logical_and(lista_clasif_predict_i, np.logical_not(lista_clasif_real_i)))
        
        #Se calcula la cantidad de Falsos Negativos (si para imagen clasif predecida = Flase y clasif real = True)
        FalseNegatives = np.sum(np.logical_and(np.logical_not(lista_clasif_predict_i), lista_clasif_real_i))
        
        #Se hacen los calculos de Precision, Recall y F1-Score
        Precision[i] = TruePositives / (TruePositives + FalsePositives)
        Recall[i] = TruePositives / (TruePositives + FalseNegatives)
        F1score[i] = 2 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i])

    print("Precision : ", Precision)
    print("Recall : ", Recall)
    print("F1 - Score: ", F1score)
train()
test()