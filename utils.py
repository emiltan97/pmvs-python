from math import sqrt
import numpy as np
from numpy.linalg import pinv
from numpy.linalg.linalg import svd

def fundamentalMatrix(ref, img) : 
    center1 = ref.center 
    pmat1 = ref.pmat
    pmat2 = img.pmat
    epipole = pmat2 @ center1 
    epipole = np.array([
        [ 0,         -epipole[2], epipole[1]],
        [ epipole[2], 0,         -epipole[0]],
        [-epipole[1], epipole[0], 0]
    ])
    fmat = epipole @ pmat2 @ pinv(pmat1)

    return fmat

def distance(feat, epiline) :
    distance = (abs(
        epiline[0]*feat.x + 
        epiline[1]*feat.y + 
        epiline[2]
    )) / (sqrt(
        epiline[0]**2 + 
        epiline[1]**2
    ))

    return distance

def triangulate(f1, f2, m1, m2) : 
    u1 = f1.x
    v1 = f1.y
    u2 = f2.x
    v2 = f2.y

    Q = np.array([
        u1*m1[2] - m1[0], 
        v1*m1[2] - m1[1], 
        u2*m2[2] - m2[0], 
        v2*m2[2] - m2[1] 
    ])

    U, E, V = svd(Q) 
    V /= V[-1:, -1:]

    return V[3, :]

def insertionSort(A) : 
    i = 1 
    while i < len(A) : 
        j = i 
        while j > 0 and A[j-1].depth > A[j].depth : 
            temp   = A[j] 
            A[j]   = A[j-1]
            A[j-1] = temp
            j = j - 1 
        i = i + 1 