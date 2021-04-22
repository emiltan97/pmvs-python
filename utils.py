import cv2 as cv
from classes import Patch
from math import sqrt
import numpy as np
from numpy.linalg import pinv
from numpy.linalg.linalg import inv, norm, svd
from numpy import dot
import optim
import os

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

def removeFeatures(image, cell) : 
    for feat1 in image.feats : 
        for feat2 in cell.feats : 
            if feat1 == feat2 : 
                image.feats.remove(feat1)

def getImage(id, images) : 
    for image in images : 
        if id == image.id :
            return image

def identifyCell(cell, image, p, rho, alpha) :
    for pprime in cell.patches :
        if isNeighbour(p, pprime, rho) :
            return False
        if isDiscontinue(pprime, image, alpha) : 
            return False

    return True

def isNeighbour(p1, p2, rho) :
    cp = p1.center
    cpprime = p2.center
    np = p1.normal
    npprime = p2.normal

    res = abs(dot((cp - cpprime), np)) + abs(dot((cp - cpprime), npprime)) 

    if res < 2*rho : 
        return True
    
    return False

def isDiscontinue(patch, image, alpha) :
    if 1 - optim.computeDiscrepancy(patch.ref, image, patch, miu) < alpha : 
        return True
    return False

def computeCenter(patch, cell) :
    x = np.array([cell.center[0], cell.center[1], 1])
    R = np.array([
        [patch.ref.pmat[0][0], patch.ref.pmat[0][1], patch.ref.pmat[0][2]],
        [patch.ref.pmat[1][0], patch.ref.pmat[1][1], patch.ref.pmat[1][2]],
        [patch.ref.pmat[2][0], patch.ref.pmat[2][1], patch.ref.pmat[2][2]]
    ])
    t = np.array([
        patch.ref.pmat[0][3],
        patch.ref.pmat[1][3],
        patch.ref.pmat[2][3]
    ])
    X = inv(R) @ (x - t)
    X = np.array([X[0], X[1], X[2], 1])
    vect = X - patch.ref.center

    t = -(patch.normal @ X - patch.normal @ patch.center) / (patch.normal @ vect)

    return X + t*vect

def hasImage(image, Vp) :
    for img in Vp :
        if img.id == image.id :
            return True
    return False

def savePatch(patch, filename) : 
    file = open(filename, 'a')
    file.write(str(patch.ref.id)); file.write(" ")
    file.write(str(patch.center[0])); file.write(" ")
    file.write(str(patch.center[1])); file.write(" ")
    file.write(str(patch.center[2])); file.write(" ")
    file.write(str(patch.center[3])); file.write(" ")
    file.write(str(patch.normal[0])); file.write(" ")
    file.write(str(patch.normal[1])); file.write(" ")
    file.write(str(patch.normal[2])); file.write(" ")
    file.write(str(patch.normal[3])); file.write(" ")
    for cell in patch.cells : 
        file.write(str(cell[0])); file.write(" ")
        file.write(str(cell[1][0])); file.write(" ")
        file.write(str(cell[1][1])); file.write(" ")
    file.write("\n")    
    file.close()

def savePatches(patches, filename) : 
    file = open(filename, 'w+')
    for patch in patches :
        file.write(str(patch.ref.id)); file.write(" ")
        file.write(str(patch.center[0])); file.write(" ")
        file.write(str(patch.center[1])); file.write(" ")
        file.write(str(patch.center[2])); file.write(" ")
        file.write(str(patch.center[3])); file.write(" ")
        file.write(str(patch.normal[0])); file.write(" ")
        file.write(str(patch.normal[1])); file.write(" ")
        file.write(str(patch.normal[2])); file.write(" ")
        file.write(str(patch.normal[3])); file.write(" ")
        for cell in patch.cells : 
            file.write(str(cell[0])); file.write(" ")
            file.write(str(cell[1][0])); file.write(" ")
            file.write(str(cell[1][1])); file.write(" ")
        file.write("\n")    
    file.close()

def loadPatches(images, filename) : 
    file = open(filename, 'r')
    lines = file.readlines()
    patches = []
    for line in lines : 
        words = line.split() 
        center = np.empty(4)
        normal = np.empty(4)
        ref = getImage(int(words.pop(0)), images)
        for i in range(len(center)) :   
            center[i] = float(words.pop(0))
        for i in range(len(normal)) : 
            normal[i] = float(words.pop(0))
        patch = Patch(center, normal, ref)
        ids = []
        while words : 
            id = int(words.pop(0))
            ids.append(id)
            x = int(words.pop(0))
            y = int(words.pop(0))
            getImage(id, images).cells[x][y].patches.append(patch)
            cell = np.array([id, [x, y]])
            patch.cells.append(cell)
        VpStar = []
        while ids : 
            VpStar.append(getImage(ids.pop(0), images))            
        patch.VpStar = VpStar
        patches.append(patch)
    file.close()
    
    return patches

def getColor(patch) :
    ref = patch.ref 
    center = patch.center 
    pmat = ref.pmat 
    img = cv.imread(ref.name)
    coord = pmat @ center
    coord /= coord[2]
    
    return img[int(coord[1])][int(coord[0])]

def writePly(patches, filename) :
    file = open(filename, 'w+')
    file.write("ply\n")
    file.write("format ascii 1.0\n")
    file.write(f'element vertex {len(patches)}\n')
    file.write("property float x\n")
    file.write("property float y\n")
    file.write("property float z\n")
    file.write("property float nx\n")
    file.write("property float ny\n")
    file.write("property float nz\n")
    file.write("property uchar diffuse_red\n")
    file.write("property uchar diffuse_green\n")
    file.write("property uchar diffuse_blue\n")
    file.write("end_header\n")
    for patch in patches : 
        file.write(str(patch.center[0])); file.write(" ")
        file.write(str(patch.center[1])); file.write(" ")
        file.write(str(patch.center[2])); file.write(" ")
        file.write(str(patch.normal[0])); file.write(" ")
        file.write(str(patch.normal[1])); file.write(" ")
        file.write(str(patch.normal[2])); file.write(" ")
        color = getColor(patch)
        file.write(str(color[0])); file.write(" ")
        file.write(str(color[1])); file.write(" ")
        file.write(str(color[2])); file.write(" ")
        file.write("\n")
