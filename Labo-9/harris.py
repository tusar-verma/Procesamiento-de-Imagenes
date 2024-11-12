import matplotlib.pyplot as plt
import skimage.io as io
from skimage import feature
import numpy as np
import os
import PIL.Image as pill
from skimage import filters
from skimage.feature import peak_local_max
from scipy import ndimage
import cv2 as cv

def gaussianon(n :int, sigma : int = 1 ) -> np.typing.NDArray[float]:
    res = np.zeros((n,n))
    centro = [(n-1)//2, (n-1)//2]
    for i in range(n):
        for j in range(n):
            res[i,j] = np.exp(-((i-centro[0])**2 + (j-centro[1])**2)/(2*sigma**2))
    return res

def gradiente(imagen : np.typing.NDArray[float]) -> tuple[np.typing.NDArray[float], np.typing.NDArray[float]]:
    dx = filters.sobel_h(imagen)
    dy = filters.sobel_v(imagen)
    return dx, dy

def harrysVentana(centro:tuple[int,int], radio:int, Ix: np.typing.NDArray[float], Iy:np.typing.NDArray[float]) ->  np.typing.NDArray[float]:
    k = 0.04
    m = Ix.shape[0]
    n = Ix.shape[1]
    Ixpad = np.zeros((m+2*radio,n+2*radio))
    Ixpad[radio:m+radio,radio:n+radio] = Ix
    Iypad = np.zeros((m+2*radio,n+2*radio))
    Iypad[radio:m+radio,radio:n+radio] = Iy
    nuevoCentro = (centro[0]+radio, centro[1]+radio)
    vx = np.arange(-radio+nuevoCentro[0],radio+nuevoCentro[0]+1)
    vy = np.arange(-radio+nuevoCentro[1],radio+nuevoCentro[1]+1)
    #print(vx,vx.size, Ixpad.shape, centro, m,n,radio, Ixpad[266,266])
    ventanaGauss = gaussianon(2*radio+1)
    ventanaDx = Ixpad[np.ix_(vx, vy)] 
    ventanaDy = Iypad[np.ix_(vx, vy)] 
    dx2 = ventanaDx**2 * ventanaGauss
    dy2 = ventanaDy**2 * ventanaGauss
    dxdy = ventanaDx*ventanaDy * ventanaGauss
    sumadx2 = np.sum(dx2)
    sumady2 = np.sum(dy2)
    sumadxdy = np.sum(dxdy)
    mres = np.array([
        [sumadx2, sumadxdy],
        [sumadxdy, sumady2]
    ])
    R = np.linalg.det(mres) - k*np.trace(mres)
    return R

def Harrys(imagen, puntosDeInteres, umbral):
    #Tal vez se puede usar vectorize
    esquinas = []
    valores = []
    dx, dy = gradiente(imagen)
    for p in puntosDeInteres:
        val = harrysVentana(p,5,dx,dy)
        if val >= umbral:
            esquinas.append(p)
            valores.append(val)
    
    valores = np.array(valores)
    fil, col = zip(*esquinas)
    im2 = np.zeros(imagen.shape)
    im2[fil,col] = valores
    esquinas = peak_local_max(im2,min_distance=5)
    return esquinas

def procesadoHarrys(imagen, umbral):
    bordes = feature.canny(imagen, sigma=3)
    puntosDeInteres = list(zip(*np.where(bordes == 1)))
    esquinas = Harrys(imagen, puntosDeInteres, umbral)
    return esquinas

def test():
    #Utilice la imagen test.png de las imagenes de prueba.
    imagenFilename = os.path.join("./Imagenes", 'test.png')
    imagenPrueba = io.imread(imagenFilename, as_gray=True)

    #Encontramos los puntos de bordes de la imagen
    bordes = feature.canny(imagenPrueba, sigma=3)

    #Creamos los pares de indices donde se encuentran
    puntosDeInteres = list(zip(*np.where(bordes == 1)))

    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    ax = axes.ravel()

    ax[0].imshow(imagenPrueba, cmap=plt.cm.gray)
    ax[1].imshow(bordes, cmap=plt.cm.gray)
    ax[2].imshow(bordes, cmap=plt.cm.gray)

    #Encontramos los puntos esquinas
    esquinas = Harrys(imagenPrueba, puntosDeInteres, 0.7)

    # tuplas (y,x)
    for punto in esquinas:
        ax[2].plot(punto[1],punto[0], 'bo',  markersize=2)

    plt.show()

def urbanCorner():
    umbral = 0
    cant_imagenes = 3
    fig, axes = plt.subplots(cant_imagenes, 4, figsize=(10, 10))
    for i in range(cant_imagenes):
        imagenFN = os.path.join("./Imagenes/Urban_Corner dataset/Urban_Corner datasets/Images", str(i+1)+'.png')
        imagen = io.imread(imagenFN, as_gray=True)
        esquinasPropias = procesadoHarrys(imagen,umbral)
        #param: imagen (en escala gris y tipo float32), tamaño ventana, tamaño sobel, k=0.04
        imagen32 = np.float32(imagen)
        valoresCv = cv.cornerHarris(imagen32,10,3,0.04)
        esquinasCv = np.argwhere(valoresCv >= 0.01*valoresCv.max())
        esquinasReales = np.loadtxt("./Imagenes/Urban_Corner dataset/Urban_Corner datasets/Ground Truth/" + str(i+1)+".txt", dtype=int)
        axes[i][0].imshow(imagen)
        axes[i][1].imshow(imagen)
        axes[i][2].imshow(imagen)
        axes[i][3].imshow(imagen)
        for punto in esquinasPropias:
            axes[i][1].plot(punto[1],punto[0], 'bo',  markersize=2)
        for punto in esquinasCv:
            axes[i][2].plot(punto[1],punto[0], 'bo',  markersize=2)
        for punto in esquinasReales:
            axes[i][3].plot(punto[1],punto[0], 'bo',  markersize=2)
    plt.show()
urbanCorner()
"""
Para la imagen primero Canny para bordes
Se OBTIENEN DERIVADAS USUALES
Luego para cada pixel borde se elige una caja
Ix es la suma se los dx de la caja punderados con una matriz gaussiana del tamaño de la caja
"""