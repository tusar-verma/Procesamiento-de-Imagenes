import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import skimage.io as io

import os

clf = svm.SVC()

path_comun = "./Imagenes/hog/"
def train():
    lista_train = []
    lista_clasif = []
    melementos = np.loadtxt(path_comun+"train/tabla.txt", dtype=int)
    for fila in melementos:
        lista_clasif += [fila[2] for i in range(fila[1]-fila[0] + 1)]
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
        lista_train.append(fd)
    clf.fit(lista_train, lista_clasif)
def test():
    lista_clasif_real = []
    lista_a_testear = []
    melementos = np.loadtxt(path_comun+"test/tabla.txt", dtype=int)
    for fila in melementos:
        lista_clasif_real += [fila[2] for i in range(fila[1]-fila[0] + 1)]
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
    lista_clasif_predict = clf.predict(lista_a_testear)
    
    cant_tipos_distintos = melementos.shape[0]
    Precision = np.zeros(cant_tipos_distintos)
    Recall = np.zeros(cant_tipos_distintos)
    F1score = np.zeros(cant_tipos_distintos)
    for i in range(cant_tipos_distintos):
        lista_clasif_real_i = lista_clasif_real == i
        lista_clasif_predict_i = lista_clasif_predict == i
        TruePositives = np.sum(np.logical_and(lista_clasif_predict_i, lista_clasif_real_i))
        FalsePositives = np.sum(np.logical_and(lista_clasif_predict_i, np.logical_not(lista_clasif_real_i)))
        FalseNegatives = np.sum(np.logical_and(np.logical_not(lista_clasif_predict_i), lista_clasif_real_i))
        Precision[i] = TruePositives / (TruePositives + FalsePositives)
        Recall[i] = TruePositives / (TruePositives + FalseNegatives)
        F1score[i] = 2 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i])

    print("Precision : ", Precision)
    print("Recall : ", Recall)
    print("F1 - Score", F1score)
train()
test()