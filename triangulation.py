import numpy as np

from numpy.linalg import svd
from numpy.linalg.linalg import inv

def myVersion(f1, f2, m1, m2) : 
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
    # if V[-1:, -1:] < 0 : 
    #     V = -1 * V 

    return V[3, :]
    # return V[:,3] 

def yasuVersion(f1, f2, m1, m2) : 
    u1 = f1.x
    v1 = f1.y
    u2 = f2.x
    v2 = f2.y

    A = np.array([
        m1[0] - u1*m1[2], 
        m1[1] - v1*m1[2], 
        m2[0] - u2*m2[2], 
        m2[1] - v2*m2[2]
    ])

    b = np.array([
        (f1.x*m1[2][3] - m1[0][3]),
        (f1.y*m1[2][3] - m1[1][3]),
        (f2.x*m2[2][3] - m2[0][3]),
        (f2.y*m2[2][3] - m2[1][3])
    ])

    AT = A.T 
    ATA = AT @ A 
    ATb = AT @ b 

    ATA3 = np.array([
        [ATA[0][0], ATA[0][1], ATA[0][2]],
        [ATA[1][0], ATA[1][1], ATA[1][2]],
        [ATA[2][0], ATA[2][1], ATA[2][2]]
    ])
    ATb3 = np.array([
        ATb[0], ATb[1], ATb[2]
    ])

    iATA3 = inv(ATA3)
    ans = iATA3 @ ATb3

    cood = np.array([
        ans[0], 
        ans[1],
        ans[2],
        1
    ])

    return cood