import numpy as np
import cv2 as cv
import harris as h
from skimage import feature
from skimage import img_as_float
import matplotlib.pyplot as plt
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

def simpleProductHomography(H,p):
    puntoQl = productHomography(H,[p])
    return puntoQl[0]

#Devuelva una array H donde H[i] = [x,y] I[x,y] es una esquina
def obtener_esquinas(imagen):
    bordes = feature.canny(imagen, sigma=3)
    puntosDeInteres = list(zip(*np.where(bordes == 1)))
    esquinas = h.Harrys(imagen,puntosDeInteres,0.001)
    return esquinas

def promedio(punto , imagen, radio):

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
    radio = 1
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

            if(coef_corr > 0.8):
                puntos_con_mayor_corr = puntos_con_mayor_corr.append([esquina_p,esquina_q])
    
    #puntos_con_mayor_corr = np.delete(puntos_con_mayor_corr,0,0)
    puntos_con_mayor_corr = np.array(puntos_con_mayor_corr)
    return puntos_con_mayor_corr

def ransac(imagen1, imagen2):
    imagen1bw = cv.cvtColor(imagen1, cv.COLOR_BGR2GRAY)
    imagen2bw = cv.cvtColor(imagen2, cv.COLOR_BGR2GRAY)
    #Detección esquinas
    esquinasP = obtener_esquinas(imagen1bw)
    esquinasQ = obtener_esquinas(imagen2bw)

    #Se filtran las esquinas, quedandonos con los pares con mayor correlación (forma q, p)
    esquinasFiltradas = correlacion(esquinasP, esquinasQ, imagen1bw, imagen2bw)


    #Se obtienen los puntos de partida P y los puntos de llegada Q en forma de lista
    puntosP = esquinasFiltradas[:,0]
    puntosQ = esquinasFiltradas[:,1]

    tolerancia = 10**(-3) #Tolerancia con la que se considera que una Homografía es consistente para un par de puntos
    minConsistentes = 0.8*puntosP.shape[0] #Mínima cantidad de esquinas que deben ser consistentes con una homografía
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

def warping(imagen1, imagen2, H):
    #la imagen 2 se considerará la perspectiva de referencia
    esquinas = np.array([[0,0], [0,imagen1.shape[1]-1], [imagen1.shape[0]-1,0], [imagen1.shape[0]-1, imagen1.shape[1]-1]])
    esquinasn = productHomography(H, esquinas)
    minfila = np.min(esquinasn[:,0])
    maxfila = np.max(esquinasn[:,0])
    mincol = np.min(esquinasn[:,1])
    maxcol = np.max(esquinasn[:,1])
    #ASUMO QUE LAS IMAGENES SE SUPERPONEN TANTO EN FILAS COMO COLUMNAS
    #En consecuencia maxcol y maxfila han de ser positivos
    offsetfila = -max(0,minfila)
    offsetcol = -max(0,mincol)
    m = max(maxfila + offsetfila, imagen2.shape[0])
    n = max(maxcol + offsetcol, imagen2.shape[1])

    
    or1 = np.array([0+offsetfila, 0+offsetcol])
    or2 = np.array([minfila+offsetfila, mincol+offsetcol])
    
    return or1, or2, m, n 


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
    capaAlpha1 = filalpha(imagen1.shape[0],imagen1.shape[1])
    capaAlpha2 = filalpha(imagen2.shape[0], imagen2.shape[1])
    capas = imagen1.shape[2]

    imagen1a = imagen1
    imagen2a = imagen2
    for i in range(capas):
        imagen1a[:,:,i] = imagen1a[:,:,i] * capaAlpha1  
        imagen2a[:,:,i] = imagen2a[:,:,i] * capaAlpha2  

    
    capaAlphaTot = np.ones((m,n))
    capaAlphaTot[or1[0]:or1[0]+m, or1[1]:or1[1]+n] = 0
    capaAlphaTot[or2[0]:or2[0]+m, or2[1]:or2[1]+n] = 0
    capaAlphaTot[or1[0]:or1[0]+m, or1[1]:or1[1]+n] += capaAlpha1
    capaAlphaTot[or2[0]:or2[0]+m, or2[1]:or2[1]+n] += capaAlpha2

    imagenTot = np.zeros((m,n,3))
    imagenTot[or1[0]:or1[0]+m, or1[1]:or1[1]+n, :] += imagen1a
    imagenTot[or2[0]:or2[0]+m, or2[1]:or2[1]+n, :] += imagen2a

    for i in range(capas):
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
    imagen1 = cv.imread("./der.png", cv.IMREAD_COLOR)
    imagen1 = img_as_float(imagen1)
    imagen2 = cv.imread("./izq.png", cv.IMREAD_COLOR)
    imagen2 = img_as_float(imagen2)

    H = ransac(imagen1, imagen2)

    or1, or2, m, n = warping(imagen1, imagen2, H)

    imagenRes = blend(imagen1, or1, imagen2, or2, m, n)

    fig, ax = plt.subplots(1,3, shape=(10,10))
    ax[0].imshow(imagen1)
    ax[1].imshow(imagen2)
    ax[2].imshow(imagenRes)
    plt.show()
    pass

main()