import cv2 as cv 
import numpy as np

from math import sin, cos, acos, asin

def estimateProjectionMatrix(x, y, P) : 
    Q = np.array([])

    for i in range(10) : 
        Pk      = P[:, i]
        zeroMat = np.zeros(4)
        xk      = x[i]
        yk      = y[i] 
        Qk      = np.array([
            np.hstack((Pk.T, zeroMat, (-xk) * (Pk.T))),
            np.hstack((zeroMat, Pk.T, (-yk) * (Pk.T)))
        ])
        if np.array_equal(Q, []) : 
            Q = Qk
        else : 
            Q = np.vstack((Q, Qk))

    U, E, V = np.linalg.svd(Q)

    res = np.array([
        [V[11, 0], V[11, 1], V[11, 2],  V[11, 3]],
        [V[11, 4], V[11, 5], V[11, 6],  V[11, 7]],
        [V[11, 8], V[11, 9], V[11, 10], V[11, 11]]
    ])

    return res 

def estimateInsExParams(M) :
    a1    = M[0, :-1]
    a2    = M[1, :-1]
    a3    = M[2, :-1]
    rho   = 1 / np.linalg.norm(a3)
    r3    = rho * a3 
    u0    = (rho**2) * (np.dot(a1, a3))
    v0    = (rho**2) * (np.dot(a2, a3))
    theta = acos(-((np.dot(np.cross(a1, a3), np.cross(a2, a3))) / ((np.linalg.norm(np.cross(a1, a3))) * (np.linalg.norm(np.cross(a2, a3))))))
    alpha = (rho**2)*(np.linalg.norm(np.cross(a1, a3))) * sin(theta) 
    beta  = (rho**2)*(np.linalg.norm(np.cross(a2, a3))) * sin(theta)
    r1    = (1/(np.linalg.norm(np.cross(a2,a3)))) * (np.cross(a2,a3))
    r2    = np.cross(r3, r1)
    b     = M[:, -1:]
    K     = np.array([
        np.hstack((alpha, -alpha*(cos(theta)/sin(theta)), u0)), 
        np.hstack((0,      beta/sin(theta),  v0)), 
        np.hstack((0,      0,                1))
    ])
    t     = rho * np.linalg.inv(K) @ b
    R     = np.array([
        r1.T,
        r2.T,
        r3.T
    ])

    return K, R, t

def estimateTriangulatedPoints(xl, yl, xr, yr, ml, mr, n) : 
    res = np.array([])

    for i in range(n) : 
        ul = xl[i]
        vl = yl[i] 
        ur = xr[i]
        vr = yr[i]

        Q = np.array([
            ul*ml[2] - ml[0],
            vl*ml[2] - ml[1],
            ur*mr[2] - mr[0],
            vr*mr[2] - mr[1]
        ])
        U, E, V = np.linalg.svd(Q)
        if V[-1:, -1:] < 0 :
            V = -1 * V
        if np.array_equal(res, []) : 
            res = V[-1:]
        else : 
            res = np.vstack((res, V[-1:]))

    return res

if __name__ == "__main__" : 
    left  = cv.imread("data/l3.png")
    right = cv.imread("data/r3.png")

    xl = np.array([
        [213],
        [242],
        [389],
        [385],
        [457],
        [455],
        [329],
        [326],
        [243],
        [240]
    ])
    yl = np.array([
        [442],
        [475],
        [515],
        [633],
        [663],
        [773],
        [692],
        [818],
        [520],
        [659]
    ])
    xr = np.array([
        [264],
        [290],
        [446],
        [441],
        [520],
        [517],
        [399],
        [396],
        [309],
        [305]
    ])
    yr = np.array([
        [456],
        [488],
        [529],
        [648],
        [678],
        [792],
        [701],
        [833],
        [531],
        [671]
    ])

    P = np.array([
        [ 0.021,  0.042,  0.084,  0.084,  0.084,  0.084, -0.002, -0.002, -0.017, -0.017],
        [-0.021, -0.042, -0.063, -0.136, -0.136, -0.198, -0.136, -0.198, -0.063, -0.136],
        [ 0,      0,      0.109,  0.109,  0.168,  0.168,  0.168,  0.168,  0.109,  0.109],
        [ 1,      1,      1,      1,      1,      1,      1,      1,      1,      1],
    ])

    ml = estimateProjectionMatrix(xl, yl, P) 
    mr = estimateProjectionMatrix(xr, yr, P) 

    kl, rl, tl = estimateInsExParams(ml)
    kr, rr, tr = estimateInsExParams(mr)

    print(rl)
    print(tl)

    triangulatedPoints = estimateTriangulatedPoints(xl, yl, xr, yr, ml, mr, 10)