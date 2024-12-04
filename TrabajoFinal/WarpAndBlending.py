import numpy as np
import cv2 as cv
import harris as h
from skimage import feature
from skimage import img_as_float
import matplotlib.pyplot as plt
#Se busca resolver Bh = 0, si no se puede se busca minimizar ||Bh||^2
#Imponemos que ||h|| = 1, se elimina así un grado de libertad


def getHomography(puntosP, puntosQ):
    """
    Recibe dos arrays de puntos de las mismas dimensiones
    Devuelve la matriz de homografia
    ## Parameters
    :puntosP: Array de puntos 
    :puntosQ: Array de puntos

    ## Returns: 
    Matriz homografica tal que q = Hp
    """
    B = []
    
    #Se añade para cada punto las dos ecuaciones asociadas al sistema dado por la matriz B
    
    for i in range(puntosP.shape[0]):
        q = puntosP[i]
        p = puntosQ[i]
        f1 = [p[0], p[1], 1, 0, 0, 0, -p[0]*q[0], -p[1]*q[0], -q[0]]
        f2 = [0, 0, 0, p[0], p[1], 1, -p[0]*q[1], -p[1]*q[1], -q[1]]
        B.append(f1)
        B.append(f2)
    
    B = np.array(B)

    #Descomponsición SVD de B, B=UDVt
    U, d, Vt = np.linalg.svd(B.T@B)

    #Buscamos el indice correspondiente al menor elemento de d 
    indMin = np.argmin(d)

    #Este indice corresponde a la mejor solución de norma 1 del sistema
    #Así se busca el autovector de AtA=V(D^2)Vt (D^2 es abuso de notación) tal que su autovalor asociado sea lo más cercano posible a 0
    h = Vt[indMin]

    #Se recupera la matriz
    H = h.reshape((3,3))/h[8]

    return H


def productHomography(H, p):
    """
    Calcula el producto de una lista de puntos y la matriz homografica
    ## Parameters
    :H: Matriz Homografica
    :p: Lista de pares de puntos (2D)

    ## Returns: 
    Lista de pares de puntos (2D) estimados por el producto de H y p
    """
    puntosP3D = np.ones((3,p.shape[0]))
    puntosP3D[:2,:] = p.T
    puntosQEstimados3D = H@puntosP3D

    puntosQEstimados = puntosQEstimados3D[:2,:]/puntosQEstimados3D[2,:]
    puntosQEstimados = puntosQEstimados.T
    return puntosQEstimados

def simpleProductHomography(H,p):
    """
    Calcula el producto entre la matriz H y el punto p
    ## Parameters
    :H: Matriz Homografica
    :p: Punto (2D)

    ## Returns: 
    Punto (2D)
    """
    puntoQl = productHomography(H,np.array([p]))
    return puntoQl[0]

#Devuelva una array H donde H[i] = [x,y] I[x,y] es una esquina
def obtener_esquinas(imagen):
    """
    Obtiene las esquinas de una imagen usando Harris

    ## Returns: 
    Lista de indices que son puntos esquina en la imagen
    """

    bordes = feature.canny(imagen, sigma=3)
    puntosDeInteres = list(zip(*np.where(bordes == 1)))
    esquinas = h.Harrys(imagen,puntosDeInteres,0.001)
    return esquinas

def promedio(punto , imagen, radio):
    """
    Computa el promedio para una ventana alrededor de un punto
    ## Parameters
    :punto: Indice del centro de la ventana (2D)
    :imagen: Imagen sobre la cual se encuentra el punto
    :radio: Radio de la ventana

    ## Returns: 
    Valor promedio de la ventana
    """
    #ventana_f = imagen[punto[0]-1:punto[0]+2]

    #ventana_c = [ventana_f[0,punto[1]-1 : punto[1]+2],ventana_f[1,punto[1]-1 : punto[1]+2],ventana_f[2,punto[1]-1 : punto[1]+2]]
    ventana = imagen[punto[0]-radio:punto[0]+radio+1, punto[1]-radio:punto[1]+radio+1]
    sum = 0
    sum = np.sum(ventana)
    #for i in range(0,3):
    #    for j in range(0,3):
    #        sum = sum + ventana_c[i][j]
    
    sum = sum / (ventana.shape[0]*ventana.shape[1])

    return sum 

def correlacion(esquinas_p, esquinas_q, imagen_p, imagen_q):
    radio = 3
    puntos_con_mayor_corr = []
   
    for i in range(esquinas_p.shape[0]):
        esquina_p = esquinas_p[i]
        for j in range(esquinas_q.shape[0]):
            
            esquina_q = esquinas_q[j]
            promedio_p = promedio(esquinas_p[i],imagen_p, radio)
            promedio_q = promedio(esquinas_q[j],imagen_q, radio)

            #ventana_f_p = imagen_p[esquina_p[0]-1 : esquina_p[0]+2]
            #ventana_f_q = imagen_q[esquina_q[0]-1 : esquina_q[0]+2]
            #ventana_c_p = [ventana_f_p[0,esquina_p[1]-1 : esquina_p[1]+2],ventana_f_p[1,esquina_p[1]-1 : esquina_p[1]+2],ventana_f_p[2,esquina_p[1]-1 : esquina_p[1]+2]]
            #ventana_c_q = [ventana_f_q[0,esquina_q[1]-1 : esquina_q[1]+2],ventana_f_q[1,esquina_q[1]-1 : esquina_q[1]+2],ventana_f_q[2,esquina_q[1]-1 : esquina_q[1]+2]]

            ventana_c_p = imagen_p[esquina_p[0]-radio : esquina_p[0]+radio+1, esquina_p[1]-radio : esquina_p[1]+radio+1]
            ventana_c_q = imagen_q[esquina_q[0]-radio : esquina_q[0]+radio+1, esquina_q[1]-radio : esquina_q[1]+radio+1]

            coef_corr_nom = 0
            coef_corr_den_p = 0
            coef_corr_den_q = 0
            for k in range(ventana_c_q.shape[0]):
                for l in range(ventana_c_q.shape[1]):
                  coef_corr_nom   = coef_corr_nom + ((ventana_c_p[k][l] - promedio_p) * (ventana_c_q[k][l] - promedio_q)) 
                  coef_corr_den_p = coef_corr_den_p + np.power((ventana_c_p[k][l] - promedio_p),2)
                  coef_corr_den_q = coef_corr_den_q + np.power((ventana_c_q[k][l] - promedio_q),2)     

            coef_corr = coef_corr_nom/np.sqrt(coef_corr_den_p * coef_corr_den_q)

            if(coef_corr > 0.9):
                puntos_con_mayor_corr.append([esquina_p,esquina_q])
    
    #puntos_con_mayor_corr = np.delete(puntos_con_mayor_corr,0,0)
    puntos_con_mayor_corr = np.array(puntos_con_mayor_corr)
    
    return puntos_con_mayor_corr

def ransac(imagen1, imagen2):

    imagen1bw = cv.cvtColor(imagen1, cv.COLOR_BGR2GRAY)
    imagen2bw = cv.cvtColor(imagen2, cv.COLOR_BGR2GRAY)
    imagen1bw = img_as_float(imagen1bw)
    imagen2bw = img_as_float(imagen2bw)

    #Detección esquinas
    esquinasP = obtener_esquinas(imagen1bw)
    matrizEsquinasP = np.zeros(imagen1bw.shape,dtype=int)

    print("Esquinas Imagen 1")
    for index in esquinasP:
        print(index)
        matrizEsquinasP[index[0],index[1]] = 255
    plt.imshow(matrizEsquinasP,cmap="magma")
    plt.title("Esquinas Imagen 1")
    plt.show()

    matrizEsquinasQ = np.zeros(imagen2bw.shape,dtype=int)
    esquinasQ = obtener_esquinas(imagen2bw)

    print("EsquinasImagen2")
    for index in esquinasQ:
        print(index)
        matrizEsquinasQ[index[0],index[1]] = 255
    plt.imshow(matrizEsquinasQ,cmap="magma")
    plt.title("Esquinas Imagen 2")
    plt.show()

    #Se filtran las esquinas, quedandonos con los pares con mayor correlación (forma q, p)
    esquinasFiltradas = correlacion(esquinasP, esquinasQ, imagen1bw, imagen2bw)


    #Se obtienen los puntos de partida P y los puntos de llegada Q en forma de lista
    puntosP = esquinasFiltradas[:,0]
    puntosQ = esquinasFiltradas[:,1]

    puntosP = np.delete(puntosP,[0,2,3,5],axis=0)
    puntosQ = np.delete(puntosQ,[0,2,3,5],axis=0)
    print("Se han relacionado las siguientes esquinas")

    for x in range(len(puntosP)):
        print("Imagen1",puntosP[x],"Imagen2",puntosQ[x])
        print("----")
    ##Esta parte la pueden comentar para avanzar con W&B
    cvHomography = cv.findHomography(puntosP,puntosQ,cv.RANSAC,10,maxIters = 10**5)[0]
    print("Homografia OpenCv-------------")
    print(cvHomography)
    cvHomography = getHomography(puntosP,puntosQ)
    print("Homografia Nuestra-------------")
    print(cvHomography)
    return cvHomography

    tolerancia = 10 #Tolerancia con la que se considera que una Homografía es consistente para un par de puntos
    minConsistentes = 0.12*puntosP.shape[0] #Mínima cantidad de esquinas que deben ser consistentes con una homografía
    maxIter = 10**5 #Cantidad máxima de iteraciones
    cantConsistentes = 0
    H = 0
    iter = 0
    maxconsist = 0
    while(cantConsistentes < minConsistentes and iter < maxIter):
        #Se seleccionan 4 puntos y se obtiene la homografía asociada
        randInd = np.random.randint(puntosP.shape[0], size = 4) 
        puntosPsel = puntosP[randInd]
        puntosQsel = puntosQ[randInd]
        
        H = getHomography(puntosPsel, puntosQsel)
        #Se obtienen los puntos Q dados por P a traves de H
        puntosQEstimados = productHomography(H, puntosP)

        #Se calcula la cantidad de puntos considerados consistente con la tolerancia impuesta
        cantConsistentes = np.sum(np.linalg.norm(puntosQEstimados - puntosQ, ord=np.inf, axis=1) < tolerancia)
        maxconsist = max(maxconsist, cantConsistentes/puntosQEstimados.shape[0])
        iter += 1
    print(maxconsist)
    #Puntos Q estimados con el último H computado
    puntosQEstimados = productHomography(H, puntosP)
    
    #Indices de puntos consistenes con el H
    indConsistentes = np.nonzero(np.atleast_1d(np.linalg.norm(puntosQEstimados - puntosQ, ord=np.inf, axis=1) < tolerancia))[0]

    #Se recuperan puntos consistentes con el último H computado
    puntosPConsistentes = puntosP[indConsistentes]
    puntosQConsistentes = puntosQ[indConsistentes]

    print("Luego del filtrado se han relacionado las siguientes esquinas")
    for x in range(len(puntosPConsistentes)):
        print("Imagen1",puntosPConsistentes[x],"Imagen2",puntosQConsistentes[x])
        print("----")
    
    #Se computa el H que minimize B*h con la matriz B dada por puntos consistenes
    #print(indConsistentes, puntosPConsistentes.shape[0],iter)
    Hres = getHomography(puntosPConsistentes,puntosQConsistentes)
    
    return Hres

def WarpingAndBlending(transformedImage, refImage, H):
    """
    Crea una imagen con fondo negro con las dimensiones apropiadas para que entren la imagen1 y la transformada de 
    la imagen 2. Sobre este fondo agrega solamente la transformada de la imagen 2
    ## Parameters
    :transformedImage: Imagen que sera transformada
    :refImage: Imagen de referencia
    :H: Matriz de Homografia
    ## Returns:
    Imagen/Matriz resultado con las dimensiones finales y la transformacion de transformedImage

    """
    print("Dimensiones refIamge",refImage.shape)
    filasRefImage = refImage.shape[0]
    columnasRefImage = refImage.shape[1]

    columnasTransformedImage = transformedImage.shape[1]
    filasTransformedImage = transformedImage.shape[0]

    print("Homografia", H)
    esquinasMarco = np.array([[0,0], 
                              [0,transformedImage.shape[1]-1],
                              [transformedImage.shape[0]-1,0],
                              [transformedImage.shape[0]-1, transformedImage.shape[1]-1]
                            ])
    print("Esquinas",esquinasMarco)
    esquinasMarcoTransformadas = productHomography(H, esquinasMarco).astype(int)
    print("EsquinasMarcoTransformadas",esquinasMarcoTransformadas)

    #Obtengo los indices de los bordes 
    minfila = np.min(esquinasMarcoTransformadas[:,0])
    maxfila = np.max(esquinasMarcoTransformadas[:,0])
    mincol = np.min(esquinasMarcoTransformadas[:,1])
    maxcol = np.max(esquinasMarcoTransformadas[:,1])

    #Obtengo las dimensiones de mi matriz resultado
    newMinFila = min(0,minfila)
    newMaxFila = max(filasRefImage,maxfila+1)
    newMinCol = min(0,mincol)
    newMaxCol = max(columnasRefImage,maxcol+1)

    print("NewMinFila,NewMaxFila,NewMinCol,NewMaxCol",newMinFila,newMaxFila,newMinCol,newMaxCol)


    #Alpha por indice para cada matriz
    alphaMatrixP = filalpha(filasRefImage,columnasRefImage)
    alphaMatrixQ = filalpha(filasTransformedImage,columnasTransformedImage)
    
    #Las dimensiones de mi matriz resultado
    resMatrixProyeccion = np.zeros((newMaxFila-newMinFila,newMaxCol-newMinCol,3))
    resMatrixReferencia = np.zeros((newMaxFila-newMinFila,newMaxCol-newMinCol,3))
    alphaMatrixPRes = np.zeros((newMaxFila-newMinFila,newMaxCol-newMinCol,3))
    alphaMatrixQRes = np.zeros((newMaxFila-newMinFila,newMaxCol-newMinCol,3))

    
    #Reservo esto para el blending
    overlapMatrix = np.zeros((newMaxFila-newMinFila,newMaxCol-newMinCol))

    #Ahora tengo que calcular el offset
    # Cuando mando algo a (x,y) pensando en las imagenes originales, en realidad quiero mandarlo a 
    # (newMinFila + x, newMinCol + y), teniendo cuidado cuando newMinFila es negativo
    offsetfila = - ((minfila < 0) * minfila)
    offsetcol = - ((mincol < 0) * mincol)
    print("offsetfila,offsetcolumna:",offsetfila,offsetcol)
    
    #Forward Warping
    for x in range(filasTransformedImage):
        for y in range(columnasTransformedImage):
            
            imageCoord = np.array([x,y,1])
            destCoord = H @ imageCoord.T
            destCoord/= destCoord[2]

            #indexRefImage = simpleProductHomography(H,[x,y]).astype(int) + np.array([offsetfila,offsetcol])

            indexTransformedImage = destCoord[:2].astype(int) + np.array([offsetfila,offsetcol])
            resMatrixProyeccion[indexTransformedImage[0],indexTransformedImage[1]] = transformedImage[x,y]
            overlapMatrix[indexTransformedImage[0],indexTransformedImage[1]] += 1
            alphaMatrixQRes[indexTransformedImage[0],indexTransformedImage[1]] = alphaMatrixQ[x,y]

    # return 1,1,1,1, resMatrix

    #Inverse warping
    # inverseHomography = np.linalg.inv(H)
    # for x in range(filasRes):
    #     for y in range(columnasRes):
            
    #         destCoord = np.array([x+offsetfila,y+offsetcol,1])

    #         srcCoord = inverseHomography @ destCoord.T
    #         srcCoord /= srcCoord[2]
    #         indexImage = srcCoord[:2].astype(int)
    #         if (indexImage[0] < filasRefImage and indexImage[1] < columnasRefImage):
    #             resMatrix[x,y] = refImage[indexImage[0],indexImage[1]]

    for x in range(filasRefImage):
        for y in range(columnasRefImage):
            destCoord = np.array([x+offsetfila,y+offsetcol])
            overlapMatrix[destCoord[0],destCoord[1]] += 1
            resMatrixReferencia[destCoord[0],destCoord[1]] = refImage[x,y]
            alphaMatrixPRes[destCoord[0],destCoord[1]] = alphaMatrixP[x,y]


    overlapMatrix = overlapMatrix >= 2
    plt.imshow(overlapMatrix,cmap="magma")
    plt.title("Overlapping")
    plt.show()
    #Ahora tengo las dos imágenes por separado y una máscara con los indices donde debo blendear



    resMatrix = np.zeros((newMaxFila-newMinFila,newMaxCol-newMinCol,3))
    print(resMatrix.shape,alphaMatrixP.shape,alphaMatrixQ.shape,resMatrixProyeccion.shape,resMatrixReferencia.shape)
    for x in range(overlapMatrix.shape[0]):
        for y in range(overlapMatrix.shape[1]):
            resMatrix[x,y] = (resMatrixProyeccion[x,y] + resMatrixReferencia[x,y]).astype(int)

            if overlapMatrix[x,y]:
                #print(f"[x,y]:[{x},{y}]")
                resMatrix[x,y] = np.multiply( (resMatrixReferencia[x,y] * alphaMatrixPRes[x,y]),(resMatrixProyeccion[x,y] * alphaMatrixQRes[x,y]) ).astype(int)
                resMatrix[x,y] = ( resMatrix[x,y]/(alphaMatrixPRes[x,y]+alphaMatrixQRes[x,y])  ).astype(int)
            else:
                resMatrix[x,y] = (resMatrixProyeccion[x,y] + resMatrixReferencia[x,y]).astype(int)

    return 1,1,1,1, resMatrix

    #ASUMO QUE LAS IMAGENES SE SUPERPONEN TANTO EN FILAS COMO COLUMNAS
    #En consecuencia maxcol y maxfila han de ser positivos

    m = max(maxfila +1, refImage.shape[0]) + offsetfila
    n = max(maxcol +1, refImage.shape[1]) +  offsetcol
    print(maxfila,offsetfila, maxcol,offsetcol)
    im1w = np.zeros((maxfila+offsetfila+1, maxcol+offsetcol+1, 3))

    # cv.warpPerspective(transformedImage, im1w, H, im1w.size)

    """
    for i in range(imagen1.shape[0]):
        for j in range(imagen1.shape[1]):
            index = np.array([i,j])
            nindex = simpleProductHomography(H, index).astype(int) + np.array([offsetfila, offsetcol])
            im1w[nindex[0],nindex[1]] = imagen1[i, j]
    """
    or1 = np.array([0+offsetfila, 0+offsetcol])
    or2 = np.array([minfila+offsetfila, mincol+offsetcol])
    
    return or1, or2, m, n, im1w


#Blend simple
def filalpha(m,n):
    desde = 0
    centro = ((m-1)/2, (n-1)/2)
    mv = -(1-desde)/centro[0]
    mh = -(1-desde)/centro[1]
    b = 1
    filtro = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            alpha = 0
            if (i > (m-1)/(n-1)*j and -(m-1)/(n-1)*j+(m-1) > i) or (i <= (m-1)/(n-1)*j and -(m-1)/(n-1)*j+(m-1) <= i):
                alpha = mh*np.abs(j-centro[1]) + b

            else:
                alpha = mv*np.abs(i-centro[0]) + b
                
            filtro[i,j] = alpha
    return filtro

#Se asume que los valores de la imágenes van de 0 a 1
def blend(imagen1, or1, imagen2, or2, m, n):
    """
    Devuelve la imagen resultante final con el blend entre imagen1 e imagen2
    # Parameters
    :imagen1: Imagen sobre la que se proyectó
    :or1: Origen de la Imagen1 en la matriz resultante
    :imagen2: Imagen proyectada
    :or2: Origen de la Imagen2 en la matriz resultante
    :n: Filas de la matriz resultante
    :m: Columnas de la matriz resultante
    # Returns
    Matriz con el blending de las dos imágenes
    """
    imagen1f = img_as_float(imagen1)
    imagen2f = img_as_float(imagen2)


    #Obtengo las dimensiones de im1 e im2
    dim1 = [imagen1f.shape[0],imagen1f.shape[1]]
    dim2 = [imagen2f.shape[0], imagen2f.shape[1]]

    #Genero la alpha-matriz para cada caso
    capaAlpha1 = filalpha(dim1[0], dim1[1])
    capaAlpha2 = filalpha(dim2[0], dim2[1])
    
    #La cantidad de capas a considerar (En RGB son 3)
    cantidadCapas = imagen1f.shape[2]

    imagen1a = imagen1f.copy()
    imagen2a = imagen2f.copy()

    for i in range(cantidadCapas):
        imagen1a[:,:,i] = imagen1a[:,:,i] * capaAlpha1  
        imagen2a[:,:,i] = imagen2a[:,:,i] * capaAlpha2  

    
    capaAlphaTot = np.ones((n,m))
    capaAlphaTot[or1[0]:or1[0]+dim1[0], or1[1]:or1[1]+dim1[1]] = 0
    capaAlphaTot[or2[0]:or2[0]+dim2[0], or2[1]:or2[1]+dim2[1]] = 0

    capaAlphaTot[or1[0]:or1[0]+dim1[0], or1[1]:or1[1]+dim1[1]] += capaAlpha1
    capaAlphaTot[or2[0]:or2[0]+dim2[0], or2[1]:or2[1]+dim2[1]] += capaAlpha2

    imagenTot = np.zeros((m,n,cantidadCapas
                          ))
    imagenTot[or1[0]:or1[0]+dim1[0], or1[1]:or1[1]+dim1[1], :] += imagen1a
    imagenTot[or2[0]:or2[0]+dim2[0], or2[1]:or2[1]+dim2[1], :] += imagen2a

    for i in range(cantidadCapas):
        imagenTot[:,:,i] = imagenTot[:,:,i]/capaAlphaTot
    
    
    return imagenTot


#Mira coincidencias entre dos imagenes en sus canales de color (RGB), desde el origen
def dice(imagen1, imagen2):
    m1 = imagen1.shape[0]
    n1 = imagen1.shape[1]
    m2 = imagen2.shape[0]
    n2 = imagen2.shape[1]

    M = max(m1,m2)
    N = max(n1,n2)
    
    im1padd = -np.ones((M,N,3))
    im2padd = -2*np.ones((M,N,3))

    im1padd[:m1,:n1,:] = imagen1
    im2padd[:m2,:n2,:] = imagen2

    dicecoeff = 2*np.sum(im1padd==im2padd)/(3*m1*n1 + 3*m2*n2)

    return dicecoeff





def main():
    imagen1 = cv.imread("./cuadrados_der.png", cv.IMREAD_COLOR)

    imagen2 = cv.imread("./cuadrados_izq.png", cv.IMREAD_COLOR)


    H = ransac(imagen1, imagen2)

    or1, or2, m, n, imw1 = WarpingAndBlending(imagen2, imagen1, H)

    plt.imshow(imw1)
    plt.show()
    cv.imwrite("test.png",imw1)
    #imagenRes = blend(imw1, or1, imagen2, or2, m, n)

    #fig, ax = plt.subplots(1,3, shape=(10,10))
    #ax[0].imshow(imagen1)
    #ax[1].imshow(imagen2)
    #ax[2].imshow(imagenRes)
    #plt.show()
    pass

main()
