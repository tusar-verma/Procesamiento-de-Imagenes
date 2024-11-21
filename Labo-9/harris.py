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
    #k se considera valor fijo
    k = 0.04
    m = Ix.shape[0]
    n = Ix.shape[1]
    #Se añade padding (con 0s) a la imagen para evistar problemas de rango al tomar una ventana
    Ixpad = np.zeros((m+2*radio,n+2*radio))
    Ixpad[radio:m+radio,radio:n+radio] = Ix
    Iypad = np.zeros((m+2*radio,n+2*radio))
    Iypad[radio:m+radio,radio:n+radio] = Iy

    #Se corrige el desfase por el padding y se obtiene el centro en la imagen nueva
    nuevoCentro = (centro[0]+radio, centro[1]+radio)
    vx = np.arange(-radio+nuevoCentro[0],radio+nuevoCentro[0]+1)
    vy = np.arange(-radio+nuevoCentro[1],radio+nuevoCentro[1]+1)

    #Se obtienen los pesos acorde a la ventan gaussiana
    ventanaGauss = gaussianon(2*radio+1)
    ventanaDx = Ixpad[np.ix_(vx, vy)] 
    ventanaDy = Iypad[np.ix_(vx, vy)] 

    #Se obtiene Ix^2 pesado
    dx2 = ventanaDx**2 * ventanaGauss
    #Se obtiene Iy^2 pesado
    dy2 = ventanaDy**2 * ventanaGauss
    #Se obtiene IxIy pesado
    dxdy = ventanaDx*ventanaDy * ventanaGauss
    #Se suman los valores
    sumadx2 = np.sum(dx2)
    sumady2 = np.sum(dy2)
    sumadxdy = np.sum(dxdy)
    #Se construye la matriz final
    mres = np.array([
        [sumadx2, sumadxdy],
        [sumadxdy, sumady2]
    ])
    #Se calcula R
    R = np.linalg.det(mres) - k*np.trace(mres)**2
    return R

#Se pide los puntos de interes junto con la imagen
def Harrys(imagen, puntosDeInteres, umbral=-1):
    esquinas = []
    valores = []
    todoVal = []
    #Se obtienen los gradientes de la imagen 
    dx, dy = gradiente(imagen)

    #Para cada punto de interes se obtiene el valor de R asociado
    for p in puntosDeInteres:
        val = harrysVentana(p,5,dx,dy)
        todoVal.append(val)
    todoVal = np.array(todoVal)
    #Si el umbral no se difinió por el usuario o es -1, se impone que sea el 0.01 del R máximo
    if umbral == -1:
        umbral = 0.01*np.max(todoVal)
    
    #Nos quedamos solo con los puntos que superen el umbral
    for i in range(len(puntosDeInteres)):
        val = todoVal[i]
        p = puntosDeInteres[i]
        if val > umbral:
            esquinas.append(p)
            valores.append(val)
    #Se construye una imagen con solo los puntos considerados esquina
    valores = np.array(valores)
    fil, col = zip(*esquinas)
    im2 = np.zeros(imagen.shape)
    im2[fil,col] = valores
    #Nos quedamos con los máximos locales, la min_distance afecta la presición con la que se detectan las esquinas
    esquinas = peak_local_max(im2,min_distance=4)
    
    #Se retornan las esquinas filtradas
    return esquinas

def procesadoHarrys(imagen, umbral=-1):
    #Se obtienen los bordes de la imagen
    bordes = feature.canny(imagen, sigma=1)

    #Se extrae la lista de puntos de interes que componen los bordes
    puntosDeInteres = list(zip(*np.where(bordes == 1)))

    #Se obtienen las esquinas con Harris
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
        
        #Se obtienen las esquinas con nuestro algoritmo
        esquinasPropias = procesadoHarrys(imagen)

        #Se obtienen esquinas con Harris de OpenCV
        #param: imagen (en escala gris y tipo float32), tamaño ventana, tamaño sobel, k=0.04
        imagen32 = np.float32(imagen)
        valoresCv = cv.cornerHarris(imagen32,10,3,0.04)
        esquinasCv = np.argwhere(valoresCv >= 0.01*valoresCv.max())
        #Se obtienen las esquinas reales
        esquinasReales = np.loadtxt("./Imagenes/Urban_Corner dataset/Urban_Corner datasets/Ground Truth/" + str(i+1)+".txt", dtype=int)
        
        axes[i][0].imshow(imagen)
        axes[i][1].imshow(imagen)
        axes[i][2].imshow(imagen)
        axes[i][3].imshow(imagen)

        #Se plotean las esquinas
        for punto in esquinasPropias:
            axes[i][1].plot(punto[1],punto[0], 'bo',  markersize=2)
        for punto in esquinasCv:
            axes[i][2].plot(punto[1],punto[0], 'bo',  markersize=2)
        for punto in esquinasReales:
            axes[i][3].plot(punto[1],punto[0], 'bo',  markersize=2)
    
    axes[0][0].set_title("Imagen Original")
    axes[0][1].set_title("Esquinas propias")
    axes[0][2].set_title("Esquinas de OpenCV")
    axes[0][3].set_title("Esquinas Reales")

    plt.show()
urbanCorner()
