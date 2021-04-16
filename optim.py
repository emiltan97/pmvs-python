import numpy as np
import cv2 as cv
from math import sqrt

depthVector = None

def computeDiscrepancy(ref, image, patch) : 
    grid1 = projectGrid(patch, ref.pmat, ref)
    grid2 = projectGrid(patch, image.pmat, image)
    val1 = computeGridValues(image, grid1)
    val2 = computeGridValues(image, grid2)

    # cv.waitKey(0)
    # cv.destroyAllWindows()  

    discrepancy = ncc(val1, val2)

    return discrepancy

def computeGridValues(image, grid) : 
    val = np.empty((5, 5, 3))
    img = cv.imread(image.name)
    for i in range(grid.shape[0]) : 
        for j in range(grid.shape[1]) : 
            x = int(grid[i][j][0])
            y = int(grid[i][j][1])
            if (x < 0 or y < 0 or x > 480 or y > 640) : 
                val[i][j] = np.array([0, 0, 0])
            else : 
                val[i][j] = img[x][y]

    return val

def projectGrid(patch, pmat, image) : 
    gridCoordinate = np.empty((5, 5, 3))
    margin = 2.5

    center = pmat @ patch.center
    center /= center[2]
    dx = pmat @ (patch.center + patch.px) 
    dy = pmat @ (patch.center + patch.py) 
    dx /= dx[2]
    dy /= dy[2]
    dx -= center
    dy -= center

    left = center - dx*margin + dy*margin
    for i in range(5) : 
        temp = left
        left = left - dy
        for j in range(5) : 
            gridCoordinate[i][j] = temp
            temp = temp + dx
    
    img = cv.imread(image.name)
    # cv.rectangle(img, (int(gridCoordinate[0][0][0]), int(gridCoordinate[0][0][1])), (int(gridCoordinate[4][4][0]), int(gridCoordinate[4][4][1])), (0, 255, 0), -1)
    # cv.imshow(f'{image.name}', img)

    return gridCoordinate

def ncc(val1, val2) : 
    length = 75
    m1 = 0 
    m2 = 0 
    for i in range(val1.shape[0]) : 
        for j in range(val1.shape[1]) :
            for k in range(val1.shape[2]) : 
                m1 += val1[i][j][k]
                m2 += val2[i][j][k]
    m1 /= length
    m2 /= length
    a = 0 
    b = 0 
    c = 0
    for i in range(val1.shape[0]) : 
        for j in range(val1.shape[1]) :
            for k in range(val1.shape[2]) : 
                d1 = val1[i][j][k] - m1
                d2 = val2[i][j][k] - m2
                a += d1 * d2 
                b += d1**2 
                c += d2**2
    if b * c == 0 : 
        return 0
    res = a / sqrt(b * c)

    return res