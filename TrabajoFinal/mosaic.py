import numpy as np
import cv2 as cv
import harris as h
from skimage import feature
from skimage import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt


def productHomography(H, p):
    puntosP3D = np.ones((3,p.shape[0]))
    puntosP3D[0,:] = (p.T)[1,:]
    puntosP3D[1,:] = (p.T)[0,:]
    puntosQEstimados3D = H@puntosP3D

    puntosQEstimados = puntosQEstimados3D[:2,:]/puntosQEstimados3D[2,:]
    puntosQEstimados = puntosQEstimados.T
    puntosQEstimados = puntosQEstimados[:,[1,0]]
    return puntosQEstimados

def simpleProductHomography(H,p):
    puntoQl = productHomography(H,np.array([p]))
    return puntoQl[0]

#Devuelva una array H donde H[i] = [x,y] I[x,y] es una esquina
def obtener_esquinas(imagen):
    bordes = feature.canny(imagen, sigma=3)
    puntosDeInteres = list(zip(*np.where(bordes == 1)))
    esquinas = h.Harrys(imagen,puntosDeInteres,0.001)
    return esquinas


def correlacion(esquinas_p, esquinas_q, imagen_p, imagen_q):
    radio = 3
    puntos_con_mayor_corr = []
   
    for i in range(esquinas_p.shape[0]):
        esquina_p = esquinas_p[i]
        max_corr = 0
        mejor_esquina = np.array([0,0])
        for j in range(esquinas_q.shape[0]):
            
            esquina_q = esquinas_q[j]
            ventana_c_p = imagen_p[esquina_p[0]-radio : esquina_p[0]+radio+1, esquina_p[1]-radio : esquina_p[1]+radio+1]
            ventana_c_q = imagen_q[esquina_q[0]-radio : esquina_q[0]+radio+1, esquina_q[1]-radio : esquina_q[1]+radio+1]
            promedio_p = np.mean(ventana_c_p)
            promedio_q = np.mean(ventana_c_q)
            
            
            if ((ventana_c_p.size != ventana_c_q.size) or (ventana_c_p.size!=(2*radio+1)**2) or ventana_c_q.size != (2*radio+1)**2):
                continue
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
                if(max_corr < coef_corr):
                    max_corr = coef_corr
                    mejor_esquina = esquina_q
        
        if(max_corr > 0.90):
            puntos_con_mayor_corr.append([esquina_p,mejor_esquina])
    
    puntos_con_mayor_corr = np.array(puntos_con_mayor_corr)
    return puntos_con_mayor_corr

def visualizar_corr(imagen1, imagen2, puntos_corr):
    offset = imagen1.shape[1] + 20
    dimX = offset + imagen2.shape[1]
    dimY = max(imagen1.shape[0],imagen2.shape[0])
    metaimagen = np.zeros((dimY, dimX))
    metaimagen[:imagen1.shape[0],:imagen1.shape[1]] = imagen1
    metaimagen[:imagen2.shape[0],offset:offset+imagen2.shape[1]] = imagen2
    plt.imshow(metaimagen)
    for par in puntos_corr:
        p = par[0]
        q = par[1]
        plt.plot([p[1], q[1] + offset],[p[0], q[0]], 'bo',  linestyle="--")
    plt.show()

def getHomography(puntosP, puntosQ):
    B = []
    
    #Se añade para cada punto las dos ecuaciones asociadas al sistema dado por la matriz B
    puntosP2 = puntosP[:,[1,0]]
    puntosQ2 = puntosQ[:,[1,0]]

    for i in range(puntosP2.shape[0]):
        p = puntosP2[i]
        q = puntosQ2[i]
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
    H = h.reshape((3,3))

    #Se estabiliza de ser posible la matriz Homográfica
    if(np.abs(h[8]) > 10**(-15)):
        H = H/h[8]

    return H

def ransac(imagen1, imagen2):
    imagen1bw = cv.cvtColor(imagen1, cv.COLOR_BGR2GRAY)
    imagen2bw = cv.cvtColor(imagen2, cv.COLOR_BGR2GRAY)
    imagen1bw = img_as_float(imagen1bw)
    imagen2bw = img_as_float(imagen2bw)
    #Detección esquinas
    esquinasP = obtener_esquinas(imagen1bw)
    esquinasQ = obtener_esquinas(imagen2bw)

    #Se filtran las esquinas, quedandonos con los pares con mayor correlación (forma q, p)
    esquinasFiltradas = correlacion(esquinasP, esquinasQ, imagen1bw, imagen2bw)

    visualizar_corr(imagen1bw, imagen2bw, esquinasFiltradas)
    #Se obtienen los puntos de partida P y los puntos de llegada Q en forma de lista
    puntosP = esquinasFiltradas[:,0]
    puntosQ = esquinasFiltradas[:,1]
    
    
    tolerancia = 2 #Tolerancia con la que se considera que una Homografía es consistente para un par de puntos
    minConsistentes = 0.8*puntosP.shape[0] #Mínima cantidad de esquinas que deben ser consistentes con una homografía
    maxIter = 10**5 #Cantidad máxima de iteraciones
    cantConsistentes = 0
    H = 0
    iter = 0

    maxconsist = 0
    Hmax = 0
    while(cantConsistentes < minConsistentes and iter < maxIter):
        #Se seleccionan 4 puntos y se obtiene la homografía asociada
        randInd = np.random.randint(puntosP.shape[0], size = 4) 
        puntosPsel = puntosP[randInd]
        puntosQsel = puntosQ[randInd]

        H = getHomography(puntosPsel, puntosQsel)
        #Se obtienen los puntos Q dados por P a traves de H
        puntosQEstimados = productHomography(H, puntosP)

        #Se calcula la cantidad de puntos considerados consistente con la tolerancia impuesta
        cantConsistentes = np.sum(np.linalg.norm(puntosQEstimados - puntosQ, ord=2, axis=1) < tolerancia)
        if(maxconsist < cantConsistentes):
            maxconsist = cantConsistentes
            Hmax = H
        iter += 1
    

    #Puntos Q estimados con el último H computado
    puntosQEstimados = productHomography(Hmax, puntosP)
    
    #Indices de puntos consistenes con el H
    indConsistentes = np.nonzero(np.atleast_1d(np.linalg.norm(puntosQEstimados - puntosQ, ord=2, axis=1) < tolerancia))[0]

    #Se recuperan puntos consistentes con el último H computado
    puntosPConsistentes = puntosP[indConsistentes]
    puntosQConsistentes = puntosQ[indConsistentes]

    #Se computa el H que minimize B*h con la matriz B dada por puntos consistenes
    Hres = getHomography(puntosPConsistentes,puntosQConsistentes)

    
    puntosQEst = productHomography(Hres, puntosP)
    puntos_corr2 = np.zeros((puntosP.shape[0],2,2))
    for i in range(puntosP.shape[0]):
        puntos_corr2[i,0] = puntosP[i]
        puntos_corr2[i,1] = puntosQEst[i]
    visualizar_corr(imagen1bw, imagen2bw, puntos_corr2)

    return Hres

def warping(imagen1, imagen2, H):
    imagen1f = img_as_float(imagen1)
    #la imagen 2 se considerará la perspectiva de referencia
    esquinas = np.array([[0,0], [0,imagen1.shape[1]-1], [imagen1.shape[0]-1,0], [imagen1.shape[0]-1,imagen1.shape[1]-1]])
    esquinasn = productHomography(H, esquinas).astype(int)
    #productHomography invierte coordenadas X e Y
    minY= np.min(esquinasn[:,0])
    maxY = np.max(esquinasn[:,0])
    minX = np.min(esquinasn[:,1])
    maxX = np.max(esquinasn[:,1])
    
    #Calculamos offset en caso de que la imagen1 mapee a coordenadas negativas
    offset = np.array([(minY<0)*-minY, (minX<0)*-minX])
    ancho = max(maxX, imagen2.shape[1])
    alto = max(maxY, imagen2.shape[0])

    im1w = np.zeros((alto+offset[0],ancho+offset[1],3))

    #Se calcula una matriz de coordenadas 2d
    ies, jes = np.indices((imagen1.shape[0],imagen1.shape[1]))
    coord = np.zeros((imagen1.shape[0],imagen1.shape[1],2))
    coord[:,:,0] = ies[:,:]
    coord[:,:,1] = jes[:,:]

    #Se transforma la matriz en un lista de coordenadas
    coord = coord.reshape((imagen1.shape[0]*imagen1.shape[1],2))

    #Se calculas las nuevas coordenadas aplicando el offset
    nuevas_coord = np.round(productHomography(H,coord)) + offset

    #Se obtienen listas de coordenadas en un formato indexable en un array 2d
    listacoord_nueva = tuple(zip(*nuevas_coord.astype(int)))
    listacoord = tuple(zip(*coord.astype(int)))

    #Se obtienen los valores viejos a asignar en la nueva imagen
    valores_viejos = imagen1f[listacoord]

    #Se asignan valores de imagen1 en las coordenadas nuevas correspondeintes dadas por la matriz homografica
    im1w[listacoord_nueva] = valores_viejos

    plt.imshow(im1w)
    plt.show()
    return im1w,offset

#Crea función piramide en 3d (varía de 1 a 0)
def filalpha(m,n):
    desde = 0.001 #Evita divisiones por 0 a posteriori
    centro = ((m-1)/2, (n-1)/2)
    #Funciones que describen la variacion de 1 a desde linealmente segun eje
    #Pendiente vertical
    mv = -(1-desde)/centro[0]

    #Pendiente horizontal
    mh = -(1-desde)/centro[1]

    #Valor al origen de ambas funciones lineales
    b = 1

    filtro = np.zeros((m,n))
    #A cada coordenada se la asinga el valor descrito por la funcion que le corresponda segun 
    #en qué lado de la piramide se encuentre: mv para los triangulos norte y sur; mh para los triangulos este y oeste
    for i in range(m):
        for j in range(n):
            alpha = 0
            #Condición se cumple si la coordenada pertenece a los cuadrantes este u oeste
            if (i > (m-1)/(n-1)*j and -(m-1)/(n-1)*j+(m-1) > i) or (i <= (m-1)/(n-1)*j and -(m-1)/(n-1)*j+(m-1) <= i):
                alpha = mh*np.abs(j-centro[1]) + b

            else:
                alpha = mv*np.abs(i-centro[0]) + b
            
            #Se asigna el alpha correspondiente
            filtro[i,j] = alpha
    return filtro

def blend2(imagenref, imagen2, offset):
    imagen1a = img_as_float(imagenref)
    imagen2a = img_as_float(imagen2)

    #Se guardan las dimensiones en Y y X de las 2 imagenes
    dim1 = [imagen1a.shape[0],imagen1a.shape[1]]
    dim2 = [imagen2a.shape[0], imagen2a.shape[1]]

    #Se crean las matrices de alphas para la imagen1 (que ya fue "warpeada") y la imagen2
    capaAlpha1 = filalpha(dim1[0], dim1[1])
    capaAlpha2 = filalpha(dim2[0], dim2[1])
    
    #Se coloca imagen2 en fonde negro del tamaño de la imagen compuesta, teniendo en cuenta offset
    fnegro = np.zeros(imagen1a.shape)
    fnegro[offset[0]:dim2[0]+offset[0],offset[1]:dim2[1]+offset[1]] =  imagen2a

    #Calculamos la matriz booleana con 1 en las coordenadas donde las dos imagenes se superponen, 0 donde no 
    superpos1 = np.ones((dim1[0],dim1[1]))
    superpos2 = np.ones((dim1[0],dim1[1]))
    for i in range(3):
        superpos1 = np.logical_and(superpos1, imagen1a[:,:,i]==0)
        superpos2 = np.logical_and(superpos2, fnegro[:,:,i]==0)

    #Tratamos parte que no es común 
    filtro = np.logical_or(superpos1,superpos2)

    #res es la imagen compuesta, ponemos la seccion sin superposición de la imagen1
    res = np.zeros(imagen1a.shape)
    for i in range(3):
        res[:,:,i] = (imagen1a[:,:,i] + fnegro[:,:,i])*filtro

    #Tratamos parte con superposición, negrando el filtro
    filtro = np.logical_not(filtro)

    #En la seccion con superposición sumamos las imágenes aplicandoles los alphas correspondientes para las coordenadas de la sección
    #Se hace para la imagen1 y la imagen2
    impagensuper1 = np.zeros(imagen1a.shape)
    impagensuper2 = np.zeros(imagen1a.shape)
    for i in range(3):
        impagensuper1[:,:,i] = imagen1a[:,:,i] *capaAlpha1*filtro
        impagensuper2[offset[0]:dim2[0]+offset[0],offset[1]:dim2[1]+offset[1],i] = fnegro[offset[0]:dim2[0]+offset[0],offset[1]:dim2[1]+offset[1],i] *capaAlpha2*filtro[offset[0]:dim2[0]+offset[0],offset[1]:dim2[1]+offset[1]]

    #Sumamos la partes con superposición
    impagensuper = impagensuper1 + impagensuper2

    #Se "normaliza" la sección con superposición dividiendo por la suma de alphas de ambas imagenes en la región
    for i in range(3):
        impagensuper[offset[0]:dim2[0]+offset[0],offset[1]:dim2[1]+offset[1],i] = (impagensuper[offset[0]:dim2[0]+offset[0],offset[1]:dim2[1]+offset[1],i]/(capaAlpha1[offset[0]:dim2[0]+offset[0],offset[1]:dim2[1]+offset[1]] + capaAlpha2)) *filtro[offset[0]:dim2[0]+offset[0],offset[1]:dim2[1]+offset[1]]
    
    #Se suman las partes sin superposición y con superposición normalizada
    res = res + impagensuper

    return res

def main():
    imagen2 = cv.imread("./imagenes/Cubo-der.jpeg", cv.IMREAD_COLOR)

    imagen1 = cv.imread("./imagenes/Cubo-izq.jpeg", cv.IMREAD_COLOR)

    #Se obtiene la matriz de homografía
    print("Calculando matriz Homográfica ... ")
    H = ransac(imagen1, imagen2)
    print("--- Se obtuvo la matriz Homográfica ---")

    #Se aplica warping recuperando el offset y la imagen1 "warpeada"
    print("Haciendo warping ...")
    imw1, offset = warping(imagen1, imagen2, H)
    print("--- Warping finalizado ---")

    #Se hace el blending de las dos imagenes (teniendo en cuenta el offset del warpeo) y se obtiene la imagen compuesta
    print("Iniciando blending ...")
    imagenRes = blend2(imw1, imagen2, offset)
    print("--- Blending finalizado ---")

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(imagen1)
    ax[1].imshow(imagen2)
    ax[2].imshow(imagenRes)
    plt.show()
    
    cv.imwrite("res_pabe2.png",img_as_ubyte(imagenRes))
    

main()
