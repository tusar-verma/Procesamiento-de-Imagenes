import numpy as np

def conv_discreta(A, B):
    # Obtener las dimensiones de las dos matrices
    m1, n1 = A.shape
    m2, n2 = B.shape
    
    # matriz resultado con padding
    result = np.zeros((m1 + m2 - 1, n1 + n2 - 1))
    
    # Realizar la convolución bidimensional
    for i in range(m1):
        for j in range(n1):
            for m in range(m2):
                for n in range(n2):
                    result[i + m, j + n] += A[i, j] * B[m, n]
    
    return result


def test():
    x = np.array([[1,4,1],[2,5,3]])

    i = np.array([[0, -1, 1], [-1,4,-1], [0, -1, 0]])
    print(conv_discreta(x, i))

    ii = np.array([[1,2,3]])
    print(conv_discreta(x, ii))

    iii = np.array([[-2],[3],[-1]])
    print(conv_discreta(x, iii))

    ### verificación propiedades convolución

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[0, 1], [2, 3]])
    C = np.array([[1, 1], [1, 1]])

    # conmutatividad

    print(conv_discreta(A, B) == conv_discreta(B,A))

    # Distributiva

    print(conv_discreta(A + B, C) == conv_discreta(A, C) + conv_discreta(B, C))

    # Asociatividad

    print(conv_discreta(conv_discreta(A, B), C) ==conv_discreta(A,conv_discreta(B,C)))