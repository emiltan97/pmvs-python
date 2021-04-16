import numpy as np

from numpy.linalg import pinv, norm 
from numpy import cross
from math import sqrt

def computeFundamentalMatrix(ref, img) : 
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

def computeDistance(feat, epiline) :
    distance = (abs(
        epiline[0]*feat.x + 
        epiline[1]*feat.y + 
        epiline[2]
    )) / (sqrt(
        epiline[0]**2 + 
        epiline[1]**2
    ))

    return distance

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

def getPatchAxes(ref, patch) : 
    pmat= ref.pmat
    mzaxes = np.array([pmat[2][0], pmat[2][1], pmat[2][2]])
    mxaxes = np.array([pmat[0][0], pmat[0][1], pmat[0][2]])
    myaxes = cross(mzaxes, mxaxes)
    myaxes /= norm(myaxes)
    mxaxes = cross(myaxes, mzaxes)
    xaxe = np.array([mxaxes[0], mxaxes[1], mxaxes[2], 0])
    yaxe = np.array([myaxes[0], myaxes[1], myaxes[2], 0])
    fx = xaxe @ pmat[0]
    fy = yaxe @ pmat[1]
    fz = norm(patch.center - ref.center)
    ftmp = fx + fy
    pscale = fz / ftmp
    # pscale = 2 * 2 * fz / ftmp
    normal3 = np.array([patch.normal[0], patch.normal[1], patch.normal[2]])
    yaxis3 = cross(normal3, mxaxes)
    yaxis3 /= norm(yaxis3)
    xaxis3 = cross(yaxis3, normal3)
    
    pxaxis = np.array([xaxis3[0], xaxis3[1], xaxis3[2], 0])
    pyaxis = np.array([yaxis3[0], yaxis3[1], yaxis3[2], 0])

    pxaxis *= pscale
    pyaxis *= pscale 

    a = pmat @ (patch.center + pxaxis)
    b = pmat @ (patch.center + pyaxis)
    c = pmat @ (patch.center)
    a /= a[2]
    b /= b[2]
    c /= c[2]

    xdis = norm(a - c)
    ydis = norm(b - c)

    if xdis != 0 :
        pxaxis /= xdis 
    if ydis != 0 :
        pyaxis /= ydis

    return pxaxis, pyaxis