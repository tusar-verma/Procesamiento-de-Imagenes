import numpy as np
import cv2 as cv
import harris as h
from skimage import feature
from skimage import img_as_float
import matplotlib.pyplot as plt
from ransac2 import ransac2

def x(p):
    return p[1]
def y(p):
    return p[0]

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
    radio = 8
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
            if (ventana_c_p.size != ventana_c_q.size):
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

            if(coef_corr > 0.8):
                if(max_corr < coef_corr):
                    max_corr = coef_corr
                    mejor_esquina = esquina_q
        
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
    puntosP2 = puntosP[:,[1,0]]
    puntosP2 = np.array(puntosP2[:,np.newaxis,:], dtype=np.float32)
    puntosQ = esquinasFiltradas[:,1]
    puntosQ2 = puntosQ[:,[1,0]]
    puntosQ2 = np.array(puntosQ2[:, np.newaxis, :], dtype=np.float32)
    
    cvHomography = cv.findHomography(puntosP2,puntosQ2,cv.RANSAC,5,maxIters = 10**6)[0]
    Hres = cvHomography

    puntosQEst = productHomography(Hres, puntosP)
    puntos_corr2 = np.zeros((puntosP.shape[0],2,2))
    for i in range(puntosP.shape[0]):
        puntos_corr2[i,0] = puntosP[i]
        puntos_corr2[i,1] = puntosQEst[i]
    visualizar_corr(imagen1bw, imagen2bw, puntos_corr2)
    return Hres

def warping(imagen1, imagen2, H):
    #la imagen 2 se considerará la perspectiva de referencia
    esquinas = np.array([[0,0], [0,imagen1.shape[1]-1], [imagen1.shape[0]-1,0], [imagen1.shape[0]-1,imagen1.shape[1]-1]])
    esquinasn = productHomography(H, esquinas).astype(int)
    #productHomography invierte coordenadas X e Y
    minX= np.min(esquinasn[:,0])
    maxX = np.max(esquinasn[:,0])
    minY = np.min(esquinasn[:,1])
    maxY = np.max(esquinasn[:,1])
    print(imagen2.shape)
    print(minX,minY)
    im1w = cv.warpPerspective(imagen1, H, (maxY, maxX))
    print(im1w.shape, maxY, maxX)
    plt.imshow(im1w)
    plt.show()
    return im1w


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

def blend2(imagenref, imagen2):
    imagen1f = img_as_float(imagenref)
    imagen2f = img_as_float(imagen2)

    imagen1a = imagen1f
    dim1 = [imagen1a.shape[0],imagen1a.shape[1]]

    imagen2a = imagen2f[:dim1[0],:dim1[1]]

    dim2 = [imagen2a.shape[0], imagen2a.shape[1]]
    capaAlpha1 = filalpha(dim1[0], dim1[1])
    capaAlpha2 = filalpha(dim2[0], dim2[1])
    

    fnegro = np.zeros(imagen1a.shape)
    fnegro[:dim2[0],:dim2[1]] =  imagen2a

    superpos1 = np.ones((dim1[0],dim1[1]))
    superpos2 = np.ones((dim1[0],dim1[1]))
    for i in range(3):
        superpos1 = np.logical_and(superpos1, imagen1a[:,:,i]==0)
        superpos2 = np.logical_and(superpos2, fnegro[:,:,i]==0)

    #Tratamos parte que no es comun 
    filtro = np.logical_or(superpos1,superpos2)
    res = np.zeros(imagen1a.shape)
    for i in range(3):
        res[:,:,i] = (imagen1a[:,:,i] + fnegro[:,:,i])*filtro

    #Tratamos parte superpuesta
    filtro = np.logical_not(filtro)

    impagensuper1 = np.zeros(imagen1a.shape)
    impagensuper2 = np.zeros(imagen1a.shape)
    for i in range(3):
        impagensuper1[:,:,i] = imagen1a[:,:,i] *capaAlpha1*filtro
        impagensuper2[:dim2[0],:dim2[1],i] = fnegro[:dim2[0],:dim2[1],i] *capaAlpha2*filtro[:dim2[0],:dim2[1]]

    impagensuper = impagensuper1 + impagensuper2

    for i in range(3):
        impagensuper[:dim2[0],:dim2[1],i] = (impagensuper[:dim2[0],:dim2[1],i]/(capaAlpha1[:dim2[0],:dim2[1]] + capaAlpha2)) *filtro[:dim2[0],:dim2[1]]
    
    res = res + impagensuper
    return res

def main():
    imagen1 = cv.imread("./der.png", cv.IMREAD_COLOR)

    imagen2 = cv.imread("./izq.png", cv.IMREAD_COLOR)

    H = ransac(imagen1, imagen2)

    imw1 = warping(imagen1, imagen2, H)

    
    #imagenRes = blend(imw1, or1, imagen2, or2, m, n)
    imagenRes = blend2(imw1, imagen2)
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(imagen1)
    ax[1].imshow(imagen2)
    ax[2].imshow(imagenRes)
    plt.show()
    

main()