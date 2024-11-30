import numpy as np
import cv2 as cv

#Se busca resolver Bh = 0, si no se puede se busca minimizar ||Bh||^2
#Imponemos que ||h|| = 1, se elimina así un grado de libertad

#puntosP y puntosQ son una lista de puntos del mismo tamaño
#Retorna la matriz de homografía
def getHomography(puntosP, puntosQ):
    B = []
    
    #Se añade para cada punto las dos ecuaciones asociadas al sistema dado por la matriz B
    
    for i in range(puntosP.size):
        q = puntosP[i]
        p = puntosQ[i]
        f1 = [p[0], p[1], 1, 0, 0, 0, -p[0]*q[0], -p[1]*q[0], -q[0]]
        f2 = [0, 0, 0, p[0], p[1], 1, -p[0]*q[1], -p[1]*q[1], -q[1]]
        B.append(f1)
        B.append(f2)
    B = np.array(B)

    #Descomponsición SVD de B, B=UDVt
    U, d, Vt = np.linalg.svd(B)

    #Buscamos el indice correspondiente al menor elemento de d 
    indMin = np.argmin(d)

    #Este indice corresponde a la mejor solución de norma 1 del sistema
    #Así se busca el autovector de AtA=V(D^2)Vt (D^2 es abuso de notación) tal que su autovalor asociado sea lo más cercano posible a 0
    h = Vt[indMin]

    #Se recupera la matriz
    H = h.reshape((3,3))

    return H

#Recibe matriz Homográfica H y lista de puntos p (2d)
#Retorna pares de puntos (2d) estimados por el producto de H por p
def productHomography(H, p):
    puntosP3D = np.ones((3,p.size))
    puntosP3D[:2,:] = p.T
    puntosQEstimados3D = H@puntosP3D

    puntosQEstimados = puntosQEstimados3D[:2,:]/puntosQEstimados3D[2,:]
    puntosQEstimados = puntosQEstimados.T
    return puntosQEstimados

#Devuelva una array H donde H[i] = [x,y] I[x,y] es una esquina
def obtener_esquinas(imagen):

    harris =  cv.cornerHarris(imagen,10,3,0.04)
    harris = harris>(0.01*np.max(harris))
    
    esquinas = np.array([[0,0]])
    
    for x in range(imagen.shape[0]):
        for y in range(imagen.shape[1]):
            if(harris[x,y]):
                esquinas = np.append(esquinas,[[x,y]],0)

    esquinas = np.delete(esquinas,0,0)
    return esquinas

def correlacion(esquinas_p, esquinas_q, imagen_p, imagen_q):

    puntos_con_mayor_corr = np.array([[0,0]])

    for i in range(esquinas_p.shape):
        for j in range(esquinas_q.shape):
            promedio_p = promedio(esquinas_p[i],imagen_p)
            promedio_q = promedio(esquinas_q[j],imagen_q)

            sum_1 = (imagen_p[esquinas_p[i,0]+1,esquinas_p[i,1]+1] - promedio_p) * (imagen_q[esquinas_q[i,0]+1,esquinas_q[i,1]+1] - promedio_q)         
            sum_2 = (imagen_p[esquinas_p[i,0],esquinas_p[i,1]+1] - promedio_p)   * (imagen_q[esquinas_q[i,0],esquinas_q[i,1]+1] - promedio_q)    
            sum_3 = (imagen_p[esquinas_p[i,0]-1,esquinas_p[i,1]+1] - promedio_p) * (imagen_q[esquinas_q[i,0]-1,esquinas_q[i,1]+1] - promedio_q)
            sum_4 = (imagen_p[esquinas_p[i,0]+1,esquinas_p[i,1]] - promedio_p)   * (imagen_q[esquinas_q[i,0]+1,esquinas_q[i,1]] - promedio_q)
            sum_5 = (imagen_p[esquinas_p[i,0],esquinas_p[i,1]] - promedio_p)     * (imagen_q[esquinas_q[i,0],esquinas_q[i,1]] - promedio_q)
            sum_6 = (imagen_p[esquinas_p[i,0]-1,esquinas_p[i,1]] - promedio_p)   * (imagen_q[esquinas_q[i,0]-1,esquinas_q[i,1]] - promedio_q)
            sum_7 = (imagen_p[esquinas_p[i,0]+1,esquinas_p[i,1]-1] - promedio_p) * (imagen_q[esquinas_q[i,0]+1,esquinas_q[i,1]-1] - promedio_q)
            sum_8 = (imagen_p[esquinas_p[i,0],esquinas_p[i,1]-1] - promedio_p)   * (imagen_q[esquinas_q[i,0],esquinas_q[i,1]-1] - promedio_q)
            sum_9 = (imagen_p[esquinas_p[i,0]+1,esquinas_p[i,1]-1] - promedio_p) * (imagen_q[esquinas_q[i,0]-1,esquinas_q[i,1]-1] - promedio_q)

            sum = sum_1 + sum_2 + sum_3 + sum_4 + sum_5 + sum_6 + sum_7 + sum_8 + sum_9 

            
    return

def ransac():
    #Detección esquinas
    esquinas = []

    #Se filtran las esquinas, quedandonos con los pares con mayor correlación (forma q, p)
    esquinasFiltradas = []


    #Se obtienen los puntos de partida P y los puntos de llegada Q en forma de lista
    puntosP = np.array([])
    puntosQ = np.array([])

    tolerancia = 10**(-3) #Tolerancia con la que se considera que una Homografía es consistente para un par de puntos
    minConsistentes = 0.8*esquinas.shape[0] #Mínima cantidad de esquinas que deben ser consistentes con una homografía
    maxIter = 10**3 #Cantidad máxima de iteraciones

    cantConsistentes = 0
    H = 0
    iter = 0

    while(cantConsistentes > minConsistentes and iter < maxIter):
        #Se seleccionan 4 puntos y se obtiene la homografía asociada
        randInd = np.random.randint(puntosP.size, size = 4) 
        puntosPsel = puntosP[randInd]
        puntosQsel = puntosQ[randInd]

        H = getHomography(puntosPsel, puntosQsel)
        
        #Se obtienen los puntos Q dados por P a traves de H
        puntosQEstimados = productHomography(H, puntosP)

        #Se calcula la cantidad de puntos considerados consistente con la tolerancia impuesta
        cantConsistentes = np.sum(np.linalg.norm(puntosQEstimados - puntosQ, ord=2, axis=1) < tolerancia)

        iter += 1
    
    #Puntos Q estimados con el último H computado
    puntosQEstimados = productHomography(H, puntosP)
    
    #Indices de puntos consistenes con el H
    indConsistentes = np.nonzero(np.linalg.norm(puntosQEstimados - puntosQ, ord=2, axis=1) < tolerancia)

    #Se recuperan puntos consistentes con el último H computado
    puntosPConsistentes = puntosP[indConsistentes]
    puntosQConsistentes = puntosQ[indConsistentes]

    #Se computa el H que minimize B*h con la matriz B dada por puntos consistenes
    Hres = getHomography(puntosPConsistentes,puntosQConsistentes)
    
    return Hres

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