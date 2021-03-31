import logging
import numpy as np
import cv2 as cv

import triangulation
from dep_2 import getPatchAxes, getPixel

from numpy.linalg import pinv, norm
from math import acos, cos, pi, sqrt
from numpy import dot
from patch import Patch

def run(images) : 
    for ref in images : 
        for feat1 in ref.features : 
            # Compute features satisfying epipolar constraints
            F = computeF(ref, images, feat1)
            # Sort features
            F = sortF(ref, feat1,F)
            for feat2 in F :
                # Initialize patch
                patch = computePatch(feat1, feat2, ref) 
                # Initialize Vp and V*p
                Vp = computeVp(ref, images, 60)
                VpStar = computeVpStar(ref, patch, Vp, 0.6)
                # if len(VpStar) < gamma :
                #     continue 
                # else : 
                #     refinePatch(ref, patch, VpStar)

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
                    # dispEpiline(feat1, feat2, ref, epiline)
    
    return F

def sortF(ref, feat1, F) :
    logging.info(f'IMAGE {ref.id:02d}:Sorting epipolar features.')
    pts = []
    for feat2 in  F : 
        img = feat2.image
        projectionMatrix1 = ref.projectionMatrix
        projectionMatrix2 = img.projectionMatrix
        opticalCentre1 = ref.opticalCentre 
        # pt = triangulation.yasuVersion(feat1, feat2, projectionMatrix1, projectionMatrix2)
        pt = triangulation.myVersion(feat1, feat2, projectionMatrix1, projectionMatrix2)
        pts.append(pt)
        depth = norm(pt - opticalCentre1)
        feat2.depth = depth
    F = insertionSort(F)

    return F

def computePatch(feat1, feat2, ref) : 
    logging.info(f'IMAGE {ref.id:02d}:Constructing patch.')
    img = feat2.image
    opticalCentre = ref.opticalCentre
    projectionMatrix1 = ref.projectionMatrix
    projectionMatrix2 = img.projectionMatrix
    centre = triangulation.myVersion(feat1, feat2, projectionMatrix1, projectionMatrix2)
    normal = opticalCentre - centre 
    normal /= norm(normal)
    patch = Patch(centre, normal, ref)
    # Compute x and y vectors lying on patch
    px, py = getPatchAxes.yasuVersion(ref, patch)
    patch.px = px 
    patch.py = py

    return patch

def computeVp(ref, images, gamma) : 
    logging.info(f'IMAGE {ref.id:02d}:Computing Vp.')
    id1 = ref.id
    Vp = []
    Vp.append(ref)
    for img in images : 
        id2 = img.id
        if id1 == id2 : 
            continue
        else : 
            opticalAxis1 = np.array([
                ref.projectionMatrix[2][0], 
                ref.projectionMatrix[2][1],
                ref.projectionMatrix[2][2]
            ]) 
            opticalAxis2 = np.array([
                img.projectionMatrix[2][0], 
                img.projectionMatrix[2][1],
                img.projectionMatrix[2][2]
            ]) 
            # ray = img.opticalCentre - patch.centre
            angle = dot(opticalAxis1, opticalAxis2)
            # angle = acos(dot(ray, patch.normal) / (norm(ray)*norm(patch.normal)))
            if angle < cos(gamma * pi/180) :
                continue
            else : 
                Vp.append(img)
    
    return Vp

def computeVpStar(ref, patch, Vp, alpha): 
    logging.info(f'IMAGE {ref.id:02d}:Computing VpStar.')
    id1 = ref.id 
    VpStar = []
    for img in Vp : 
        id2 = img.id
        if id1 == id2 : 
            continue 
        else : 
            h = 1 - ncc(ref, img, patch)
            if h < alpha :
                VpStar.append(img)
    
    return VpStar

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

def dispEpiline(feat1, feat2, ref, epiline) :
    ref2 = cv.imread(ref.name)
    cv.circle(ref2, (int(feat1.x), int(feat1.y)), 4, (0, 255, 0), -1)
    cv.imshow(f'Reference Image ID : {ref.id}', ref2)
    img = feat2.image.computeFeatureMap() 
    epiline_x = (int(-epiline[2] / epiline[0]), 0)
    epiline_y = (int((-epiline[2] - (epiline[1]*480)) / epiline[0]), 480)
    cv.line(img, epiline_x, epiline_y, (255, 0, 0), 1)
    cv.circle(img, (int(feat2.x), int(feat2.y)), 3, (0, 255, 0), -1)
    cv.imshow(f'Sensed Image ID : {feat2.image.id}', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

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

def ncc(ref, img, patch) : 
    # Project the patch with grid onto each image 
    projectionMatrix1 = ref.projectionMatrix
    projectionMatrix2 = img.projectionMatrix
    gridCoordinate1 = projectGrid(patch, projectionMatrix1)
    gridCoordinate2 = projectGrid(patch, projectionMatrix2)

    gridVal1 = bilinearInterpolationModule(ref, gridCoordinate1)
    gridVal2 = bilinearInterpolationModule(img, gridCoordinate2)

    return computeNCC(gridVal1, gridVal2)
    
def projectGrid(patch, pmat) : 

    # centre = np.array([0.0732381, -0.0622898, -0.145384, 1])
    # normal = np.array([-0.769807, 0.314262, 0.555551, 0])
    # patch = Patch(centre, normal, None)
    # patch.px = np.array([ 0.000120652, 0.000489067, -0.000109471, 0])
    # patch.py = np.array([-0.000272776, -1.5366e-05, -0.000369284, 0])
    # pmat = np.array([
    #     [943.823, 3085.76, -803.971, 173.349],
    #     [1992.27, 106.377, 2668.09, 263.376],
    #     [0.808701, -0.279553, -0.517545, 0.667882],
    # ])

    gridCoordinate = np.empty((5, 5, 3))
    margin = 2.5
    # pmattmp = pmat / 2
    # pmat = np.array([pmattmp[0], pmattmp[1], pmat[2]])

    centre = pmat @ patch.centre
    centre /= centre[2]
    dx = pmat @ (patch.centre + patch.px) 
    dy = pmat @ (patch.centre + patch.py) 
    dx /= dx[2]
    dy /= dy[2]
    dx -= centre
    dy -= centre

    left = centre - dx*margin - dy*margin
    for i in range(5) : 
        temp = left
        left = left + dy
        print(left)
        for j in range(5) : 
            gridCoordinate[i][j] = temp
            temp += dx
    
    exit()

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