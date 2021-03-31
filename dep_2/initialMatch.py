import logging
import numpy as np
import cv2 as cv

import getPatchAxes
import getPixel

from numpy.core.numeric import Inf
from scipy.optimize import minimize
from patch import Patch
from numpy.linalg import pinv, svd, norm
from numpy import dot, cross
from math import atan, sqrt, pi, cos, acos, asin, atan2

def run(images, gamma) : 
    for ref in images : 
        for feat1 in ref.features : 
            # Compute features satisfying epipolar constraints
            F = computeF(ref, images, feat1)
            # Sort features
            F = sortF(ref, feat1, F)
            for feat2 in F :
                # Initialize patch
                patch = computePatch(feat1, feat2, ref) 
                # Initialize Vp and V*p
                Vp = computeVp(ref, images, patch, 60)
                VpStar = computeVpStar(ref, patch, Vp, 0.6)
                if len(VpStar) < gamma :
                    continue 
                else : 
                    refinePatch(ref, patch, VpStar)

def computeF(ref, images, feat1) : 
    logging.info(f'IMAGE {ref.id:02d}:Computing epipolar features.')
    id1 = ref.id 
    F = []
    coordinate = np.array([
        feat1.x,
        feat1.y,
        1
    ])
    logging.debug(f'Feature coordinate : {coordinate}')
    for img in images : 
        id2 = img.id
        features = img.features
        if id1 == id2 : 
            continue
        else : 
            fundamentalMatrix = computeFundamentalMatrix(ref, img)
            epiline = fundamentalMatrix @ coordinate
            logging.debug(f'Epiline : {epiline}')
            for feat2 in features : 
                dist = computeDistance(feat2, epiline)
                if dist <= 5 : 
                    F.append(feat2)
    
    return F

def dispFeatureByDepth(ref, pts, images) :
    ax = plt.axes(projection='3d')

    xdata = []
    ydata = []
    zdata = []
    xdata1 = []
    ydata2 = []
    zdata3 = []

    for image in images : 
        opX = image.opticalCentre[0]
        opY = image.opticalCentre[1]
        opZ = image.opticalCentre[2]

        xdata1.append(opX)
        ydata2.append(opY)
        zdata3.append(opZ)
    for pt in pts : 
        opX = pt[0]
        opY = pt[1]
        opZ = pt[2]

        xdata.append(opX)
        ydata.append(opY)
        zdata.append(opZ)

    ax.scatter3D(xdata1, ydata2, zdata3, 'Blue')
    ax.scatter3D(xdata, ydata, zdata, 'Red')
    ax.scatter3D(ref.opticalCentre[0], ref.opticalCentre[1], ref.opticalCentre[2], 'Green')

    plt.show()

def sortF(ref, feat1, F) :
    logging.info(f'IMAGE {ref.id:02d}:Sorting epipolar features.')
    for feat2 in  F : 
        img = feat2.image
        projectionMatrix1 = ref.projectionMatrix
        projectionMatrix2 = img.projectionMatrix
        opticalCentre1 = ref.opticalCentre 
        pt = triangulate(feat1, feat2, projectionMatrix1, projectionMatrix2)[0]
        depth = norm(pt - opticalCentre1)
        feat2.depth = depth
    F = insertionSort(F)

    return F

def computeVp(ref, images, patch, gamma) : 
    logging.info(f'IMAGE {ref.id:02d}:Computing Vp.')
    id1 = ref.id
    Vp = []
    Vp.append(ref)
    for img in images : 
        id2 = img.id
        if id1 == id2 : 
            continue
        else : 
            ray = img.opticalCentre - patch.centre
            angle = acos(dot(ray, patch.normal) / (norm(ray)*norm(patch.normal)))
            if cos(angle) < cos(gamma * pi/180) :
                continue
            else : 
                Vp.append(img)
    
    return Vp

def computeVpStar(ref, patch, Vp, alpha) : 
    logging.info(f'IMAGE {ref.id:02d}:Computing VpStar.')
    id1 = ref.id
    VpStar = []
    for img in Vp : 
        id2 = img.id
        if id1 == id2 :
            continue
        else : 
            h = 1-ncc(ref, img, patch)
            if h < alpha : 
                VpStar.append(img)
    
    return VpStar

def encode(ref, centre, normal) :
    depth = norm(ref.opticalCentre - centre)
    # normal /= normal[3]
    normal = np.array([normal[0], normal[1], normal[2]])

    x = normal[0]
    y = normal[1]
    z = normal[2]

    alpha = atan(x / (-y))
    beta = atan(sqrt((x**2 + y**2)/z))

    # zaxis = np.array([
    #     normal[0],
    #     normal[1],
    #     normal[2]
    # ])
    # xaxis = np.array([
    #     ref.extrinsic[0][0],
    #     ref.extrinsic[0][1],
    #     ref.extrinsic[0][2]
    # ])
    # yaxis = cross(zaxis, xaxis)
    # yaxis /= norm(yaxis)
    # xaxis = cross(yaxis, zaxis)

    # # normal /= normal[3]

    # fx = xaxis.dot(normal)
    # fy = yaxis.dot(normal)
    # fz = zaxis.dot(normal)

    # alpha = asin(fy)
    # cosAlpha = cos(alpha)
    # sinBeta = fx / cosAlpha
    # cosBeta = -fz / cosAlpha
    # beta = acos(cosBeta)
    # if sinBeta < 0 : 
    #     beta = -beta

    return depth, alpha, beta

def refinePatch(ref, patch, VpStar) : 
    for img in VpStar : 
        if img.id == ref.id : 
            continue
        else : 
            print(img.opticalCentre)
            exit()
    DoF = encode(ref, patch.centre, patch.normal)

    minAngle = -pi / 2
    maxAngle = pi / 2
    
    lowerBound = np.array([
        -Inf, 
        minAngle,
        minAngle
    ])
    upperBound = np.array([
        Inf, 
        maxAngle,
        maxAngle
    ])
    bound = (lowerBound, upperBound)

    result = minimize(fun=computeGStar(ref, VpStar, patch), x0=DoF, method='Nelder-Mead', bounds=bound, options={'maxfav':1000})

    print(result)
    exit()

def computeGStar(ref, VpStar, patch) : 
    gStar = 0 
    for img in VpStar : 
        if img.id == ref.id : 
            continue
        else : 
            gStar += 1 - ncc(ref, img, patch)
    gStar /= len(VpStar) - 1

    return gStar

def ncc(ref, img, patch) : 
    # Compute x and y vectors lying on patch
    px, py = getPatchAxes.mainVersion(ref, patch)
    patchGrid = applyGrid(px, py ,patch)
    
    # Project the patch with grid onto each image 
    projectionMatrix1 = ref.projectionMatrix
    projectionMatrix2 = img.projectionMatrix
    gridCoordinate1 = projectGrid(patchGrid, projectionMatrix1)
    gridCoordinate2 = projectGrid(patchGrid, projectionMatrix2)

    gridVal1 = bilinearInterpolationModule(ref, gridCoordinate1)
    gridVal2 = bilinearInterpolationModule(img, gridCoordinate2)

    return computeNCC(gridVal1, gridVal2)
    
def projectGrid(grid, pm) : 
    gridCoordinate = np.empty((5, 5, 2))
    for i in range(grid.shape[0]) :
        for j in range(grid.shape[1]) :
            pt = pm @ grid[i][j]
            pt /= pt[2]
            pt = np.array([pt[0], pt[1]])
            gridCoordinate[i][j] = pt
    
    return gridCoordinate

def applyGrid(px, py, patch) : 
    grid = np.empty((5, 5, 4))
    i = -2
    x = 0 
    while i <= 2 :
        j = -2 
        y = 0
        while j <= 2 :
            gridPt = patch.centre + i*px + j*py
            grid[x][y] = gridPt
            j += 1
            y += 1
        i += 1
        x += 1

    return grid

def computePatch(feat1, feat2, ref) : 
    logging.info(f'IMAGE {ref.id:02d}:Constructing patch.')
    img = feat2.image
    opticalCentre = ref.opticalCentre
    projectionMatrix1 = ref.projectionMatrix
    projectionMatrix2 = img.projectionMatrix
    centre = triangulate(feat1, feat2, projectionMatrix1, projectionMatrix2)[0]
    normal = opticalCentre - centre 
    normal /= norm(normal)
    patch = Patch(centre, normal, ref)

    return patch

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
    
    return A

def computeDistance(feature, epiline) : 
    distance = (abs(
        epiline[0]*feature.x + 
        epiline[1]*feature.y + 
        epiline[2]
    )) / (sqrt(
        epiline[0]**2 + 
        epiline[1]**2
    ))

    return distance

def computeFundamentalMatrix(ref, img) : 
    opticalCentre1 = ref.opticalCentre
    projectionMatrix1 = ref.projectionMatrix
    projectionMatrix2 = img.projectionMatrix
    epipole = projectionMatrix2 @ opticalCentre1
    epipole = np.array([
        [ 0,         -epipole[2], epipole[1]],
        [ epipole[2], 0,         -epipole[0]],
        [-epipole[1], epipole[0], 0]
    ])
    fundamentalMatrix = epipole @ projectionMatrix2 @ pinv(projectionMatrix1)
    fundamentalMatrix = fundamentalMatrix / fundamentalMatrix[-1, -1]
    logging.debug(f'Fundamental Matrix : {fundamentalMatrix}')

    return fundamentalMatrix

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
    if V[-1:, -1:] < 0 : 
        V = -1 * V 

    return V[-1:]

def bilinearInterpolationModule(img, grid) : 
    gridVal = np.empty((5, 5, 3))
    for i in range(grid.shape[0]) :
        for j in range(grid.shape[1]) :
            x   = grid[i][j][0]
            y   = grid[i][j][1]
            if (int(x) < 0 or int(y) < 0 or int(x) >= 640 or int(y) >= 480) : 
                gridVal[i][j] = np.array([0, 0, 0])
            else : 
                x1  = int(grid[i][j][0]) 
                x2  = int(grid[i][j][0]) + 1
                y1  = int(grid[i][j][1]) 
                y2  = int(grid[i][j][1]) + 1
                q11 = getPixel.jiaChenVersion((x1, y1), img)
                q12 = getPixel.jiaChenVersion((x1, y2), img)
                q21 = getPixel.jiaChenVersion((x2, y1), img)
                q22 = getPixel.jiaChenVersion((x2, y2), img)
                gridVal[i][j] = computeBilinearInterpolation(x, y, x1, x2, y1, y2, q11, q12, q21, q22)

    return gridVal

def computeBilinearInterpolation(x, y, x1, x2, y1, y2, q11, q12, q21, q22) : 
    t = (x-x1) / (x2-x1)    
    u = (y-y1) / (y2-y1)

    a = q11*(1-t)*(1-u)
    b = q21*(t)*(1-u)
    c = q12*(u)*(1-t)
    d = q22*(t)*(u)

    f = a + b + c + d

    return f 

def computeNCC(gridVal1, gridVal2) : 
    length = 75
    mean1 = 0 
    mean2 = 0
    for i in range(gridVal1.shape[0]) :
        for j in range(gridVal1.shape[1]) : 
            for k in range(gridVal1.shape[2]) :
                mean1 += gridVal1[i][j][k]
                mean2 += gridVal2[i][j][k]
    mean1 /= length
    mean2 /= length
    product = 0
    std1 = 0
    std2 = 0
    for i in range(gridVal1.shape[0]) :
        for j in range(gridVal1.shape[1]) : 
            for k in range(gridVal1.shape[2]) :
                diff1 = gridVal1[i][j][k] - mean1 
                diff2 = gridVal2[i][j][k] - mean2
                product += diff1 * diff2
                std1 += diff1**2
                std2 += diff2**2
    stds = std1 * std2
    if stds == 0 :
        return 0 
    else :
        return product / sqrt(stds)