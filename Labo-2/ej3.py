import numpy as np

def conv_discreta(im1, im2):
    m1, n1 = im1.shape
    m2, n2 = im2.shape
    
    cant_padding_horiz = m2//2
    cant_padding_vert = n2//2
    
    im1_padd = np.pad(im1, [(cant_padding_horiz, cant_padding_horiz), (cant_padding_vert, cant_padding_vert)])
    result = np.zeros(im1_padd.shape)
    
    # por cada pixel de la imagen
    for u in range(m1):
        for v in range(n1):
            # recorro toda la segunda imagen para calcular la conv
            # El recorrido se hace tomando el centro de la imagen como (0,0)
            for i in range(- (m2//2), m2//2 +1):
                for j in range(-(n2//2), n2//2 +1):
                    # los indices se ajustaron para tomar en cuenta el padding
                    result[u + cant_padding_horiz][v + cant_padding_vert] += im1_padd[u+ cant_padding_horiz -i][v+cant_padding_vert -j] * im2[i + cant_padding_horiz][j + cant_padding_vert]
                    
    return result