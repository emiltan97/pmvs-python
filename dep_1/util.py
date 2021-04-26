import numpy as np
import cv2 as cv
import logging

from feature import Feature
from image import Image
from patch import Patch
from cell import Cell
from numpy.linalg import inv, pinv, svd, norm
from numpy import dot, abs, cross
from math import cos, pi, sqrt, acos

def initImages(filename) : 
    imageFile = open(filename, "r")
    images    = []
    imageID   = 0
    lines     = imageFile.readlines()
    for line in lines : 
        words = line.split()
        imageName = words[0]
        intrinsic = np.array([
            [float(words[1]), float(words[2]), float(words[3])],
            [float(words[4]), float(words[5]), float(words[6])],
            [float(words[7]), float(words[8]), float(words[9])]
        ])
        extrinsic = np.array([
            [float(words[10]), float(words[11]), float(words[12]), float(words[19])],
            [float(words[13]), float(words[14]), float(words[15]), float(words[20])],
            [float(words[16]), float(words[17]), float(words[18]), float(words[21])]
        ])
        img = Image(imageName, intrinsic, extrinsic, imageID, len(lines))
        images.append(img)
        imageID += 1
    imageFile.close()
    logging.info(f'Total Images : {len(images)}')

    return images

def calibrateImages(images) : 
    for image in images : 
        logging.info(f'IMAGE {image.getImageID():02d}:Calibrating Image')
        intrinsic        = image.intrinsic
        extrinsic        = image.extrinsic
        # Compute Projection Matrix : P = K[R t]
        projectionMatrix = intrinsic @ extrinsic
        logging.debug(f'ProjectionMatrix : {projectionMatrix}')
        # Compute Optical Centre : C = -R^{-1} * t
        R                = np.array([
            [extrinsic[0][0], extrinsic[0][1], extrinsic[0][2]],
            [extrinsic[1][0], extrinsic[1][1], extrinsic[1][2]],
            [extrinsic[2][0], extrinsic[2][1], extrinsic[2][2]]
        ])
        t                = np.array([
            extrinsic[0][3],
            extrinsic[1][3],
            extrinsic[2][3]
        ])
        opticalCentre    = -inv(R) @ t
        opticalCentre    = np.array([
            opticalCentre[0],
            opticalCentre[1],
            opticalCentre[2],
            1
        ])
        logging.debug(f'Optical Centre : {opticalCentre}')
        opticalAxis      = np.array([
            projectionMatrix[2][0], 
            projectionMatrix[2][1],
            projectionMatrix[2][2]
        ]) 
        logging.debug(f'Optical Axis : {opticalAxis}')
        image.setProjectionMatrix(projectionMatrix)
        image.setOpticalCentre(opticalCentre)
        image.setOpticalAxis(opticalAxis)

def applyGrid(images, gridSize, isDisplay) : 
    for image in images : 
        cells = [] 
        logging.info(f'IMAGE {image.getImageID():02d}:Applying {gridSize} * {gridSize} pixel^2 Grid')
        img = cv.imread(image.getImageName())
        width = img.shape[0]
        height = img.shape[1]
        y = 0
        index = 0
        while y < height :
            x = 0
            while x < width :
                cell = Cell((x, y), img[y:y+gridSize, x:x+gridSize],index)
                cells.append(cell)
                x += gridSize
                index += 1
            y += gridSize
        if isDisplay :  
            drawGrid(img, gridSize)
            cv.imshow(f'Image {image.getImageID()}', img)
            cv.waitKey(0)
            cv.destroyAllWindows()


def drawGrid(img, gridSize) : 
    x = gridSize
    y = gridSize
    while x < img.shape[1] : 
        cv.line(img, (x, 0), (x, img.shape[0]), (0, 255, 0), 1)
        x += gridSize
    while y < img.shape[0] : 
        cv.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)
        y += gridSize

def HarrisCorner(images, isDisplay) : 
    for image in images : 
        features = []
        logging.info(f'IMAGE {image.getImageID():02d}:Applying Harris Corners Detection')
        imageName                     = image.getImageName()
        img                           = cv.imread(imageName)
        gray                          = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray                          = np.float32(gray)
        dst                           = cv.cornerHarris(gray, 2, 3,0.0001)
        ret, dst                      = cv.threshold(dst,0.001*dst.max(),255,0)
        dst                           = np.uint8(dst)
        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
        criteria                      = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners                       = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

        for corner in corners : 
            if isDisplay : 
                # img[dst>dst.max()]=[0,0,255]
                cv.circle(img, (int(corner[0]), int(corner[1])), 4, (0, 0, 255), -1)
            feature = Feature(corner[0], corner[1], image)
            features.append(feature) 
        if isDisplay : 
            cv.imshow(f'Image {image.getImageID()}', img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        image.setFeatures(features)

def SIFT(images, isDisplay) :
    for image in images : 
        logging.info(f'IMAGE {image.getImageID():02d}:Applying SIFT Feature Detection')
        imageName = image.getImageName()
        img       = cv.imread(imageName)
        gray      = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift      = cv.SIFT_create()
        features  = []
        keypoints = sift.detect(gray, None)
        for keypoint in keypoints : 
            coordinate = keypoint.pt
            if isDisplay : 
                cv.circle(img, (int(coordinate[0]), int(coordinate[1])), 4, (0, 0, 255), -1)
            feature    = Feature(coordinate[0], coordinate[1], image)
            features.append(feature) 
        if isDisplay : 
            # drawGrid(img, 32)
            cv.imshow(f'Image {image.getImageID()}', img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        image.setFeatures(features)

def computeNeighbourImages(referenceImage, images, isDisplay) : 
    id1                    = referenceImage.getImageID()
    opticalAxis1           = referenceImage.getOpticalAxis()
    potentialVisibleImages = [] 
    if isDisplay : 
        cv.imshow(f'Reference Image ID : {referenceImage.getImageID()}', cv.imread(referenceImage.getImageName()))
    for sensedImage in images :
        id2 = sensedImage.getImageID() 
        if id1 == id2 : 
            continue
        else : 
            opticalAxis2 = sensedImage.getOpticalAxis() 
            angle        = dot(opticalAxis1, opticalAxis2)
            logging.debug(f'Angle between Image {id1} and Image {id2} : {angle*180/pi}')
            if angle < cos(60 * pi/180) :
                continue
            else : 
                potentialVisibleImages.append(sensedImage)
                if isDisplay :
                    cv.imshow(f'Sensed Image ID : {id2}', cv.imread(sensedImage.getImageName()))
    logging.info(f'IMAGE {id1:02d}:Total Number of Potentially Possible Images : {len(potentialVisibleImages)}')
    if isDisplay :
        cv.waitKey(0)
        cv.destroyAllWindows()
    for potentialVisibleImage in potentialVisibleImages : 
        fundamentalMatrix = computeFundamentalMatrix(referenceImage, potentialVisibleImage) 
        referenceImage.setFundamentalMatrix(potentialVisibleImage.getImageID(), fundamentalMatrix)

    return potentialVisibleImages

def computePotentialFeatures(referenceImage, potentialVisibleImages, feature, isDisplay) : 
    id1 = referenceImage.getImageID() 
    potentialFeatures = []
    coordinate        = np.array([
        feature.getX(), 
        feature.getY(),
        1
    ])
    logging.debug(f'Feature Coordinate : {coordinate}')
    for potentialVisibleImage in potentialVisibleImages : 
        id2 = potentialVisibleImage.getImageID() 
        if id1 == id2 : 
            continue
        else : 
            fundamentalMatrix = referenceImage.getFundamentalMatrix(potentialVisibleImage.getImageID())
            features = potentialVisibleImage.getFeatures()
            epiline  = fundamentalMatrix @ coordinate
            logging.debug(f'Epiline : {epiline}')
            temp     = filterFeaturesByEpipolarConstraint(features, epiline, feature, isDisplay)
            for feat in temp :
                potentialFeatures.append(feat)

    return potentialFeatures    

def computePatch(feature, potentialFeature, referenceImage) : 
    sensedImage       = potentialFeature.getImage()
    opticalCentre1    = referenceImage.getOpticalCentre()
    projectionMatrix1 = referenceImage.getProjectionMatrix()
    projectionMatrix2 = sensedImage.getProjectionMatrix()
    centre            = triangulate(feature, potentialFeature, projectionMatrix1, projectionMatrix2)
    centre            = np.array([centre[0][0], centre[0][1], centre[0][2], centre[0][3]])
    normal            = opticalCentre1 - centre
    patch = Patch(centre, normal, referenceImage)

    # gridCoordinate1 = projectPatch(referenceImage.getProjectionMatrix(), patch)
    # gridCoordinate2 = projectPatch(sensedImage.getProjectionMatrix(), patch)
    # ref = cv.imread(referenceImage.getImageName())
    # fundamentalMatrix = referenceImage.getFundamentalMatrix(potentialFeature.getImage().getImageID())
    # coordinate        = np.array([
    #     feature.getX(), 
    #     feature.getY(),
    #     1
    # ])
    # epiline   = fundamentalMatrix @ coordinate
    # img = sensedImage.computeFeatureMapSatisfyingEpiline(epiline)
    # epiline_x = (int(-epiline[2] / epiline[0]), 0)
    # epiline_y = (int((-epiline[2] - (epiline[1]*480)) / epiline[0]), 480)
    # cv.line(img, epiline_x, epiline_y, (255, 0, 0), 1)
    # cv.circle(ref, (int(feature.getX()), int(feature.getY())), 4, (0, 0, 255), -1)
    # cv.rectangle(ref, (int(gridCoordinate1[0][0][0]), int(gridCoordinate1[0][0][1])), (int(gridCoordinate1[4][4][0]), int(gridCoordinate1[4][4][1])), (0, 255, 0), -1)
    # cv.rectangle(img, (int(gridCoordinate2[0][0][0]), int(gridCoordinate2[0][0][1])), (int(gridCoordinate2[4][4][0]), int(gridCoordinate2[4][4][1])), (0, 255, 0), -1)
    # cv.imshow(f'Reference Image', ref)
    # cv.imshow(f'Sensed Image', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return patch
                  
def computePotentialVisibleImages(referenceImage, potentialVisibleImages, patch, beta) :
    for sensedImage in potentialVisibleImages :
        gridCoordinate1 = projectPatch(referenceImage.getProjectionMatrix(), patch)
        gridCoordinate2 = projectPatch(sensedImage.getProjectionMatrix(), patch)

        gridVal1 = bilinearInterpolationModule(cv.imread(referenceImage.getImageName()), gridCoordinate1)
        gridVal2 = bilinearInterpolationModule(cv.imread(sensedImage.getImageName()), gridCoordinate2)            

        print(computeNCC(gridVal1, gridVal2))

def projectPatch(projectionMatrix, patch) : 
    normal = patch.getNormal()
    centre = patch.getCentre()
    inv = pinv(projectionMatrix)
    scaleR = (inv[0][0] * normal[0]) + (inv[1][0] * normal[1]) + (inv[2][0] * normal[2])
    scaleU = (inv[0][1] * normal[0]) + (inv[1][1] * normal[1]) + (inv[2][1] * normal[2])
    right = np.array([inv[0][0]-scaleR*normal[0], inv[1][0]-scaleR*normal[1], inv[2][0]-scaleR*normal[2], 0])
    up = np.array([inv[0][1]-scaleU*normal[0], inv[1][1]-scaleU*normal[1], inv[2][1]-scaleU*normal[2], 0])
    scale = dot(centre, projectionMatrix[2])
    right *= scale
    up *= scale

    projCenter = projectionMatrix @ centre 
    projRight = projectionMatrix @ right
    projUp = projectionMatrix @ up
    scale = 1 / projCenter[2]
    projRight *= scale
    projUp *= scale
    projCenter *= scale

    step = 2
    diag = projRight + projUp
    diag *= step
    tl = projCenter - diag

    # grid = np.empty([25, 2])
    # index = 0 
    grid = np.empty((5, 5, 2))
    i = 0
    for y in range(5) : 
        j = 0
        for x in range(5) : 
            pt = np.array([tl[0] + x, tl[1] + y])
            # grid[index] = pt
            # index += 1
            grid[i][j] = pt
            j += 1
        i += 1

    return grid

def filterFeaturesByEpipolarConstraint(features, epiline, referenceFeature, isDisplay) : 
    potentialFeatures = []
    for feature in features : 
        distance   = (abs(
            epiline[0]*feature.getX() + 
            epiline[1]*feature.getY() + 
            epiline[2]
        )) / (sqrt(
            epiline[0]**2 + 
            epiline[1]**2
        ))
        if distance <= 5 : 
            potentialFeatures.append(feature)
    if isDisplay : 
        for feature in potentialFeatures : 
            ref = cv.imread(referenceFeature.getImage().getImageName())
            cv.circle(ref, (int(referenceFeature.getX()), int(referenceFeature.getY())), 4, (0, 255, 0), -1)
            cv.imshow(f'Reference Image ID : {referenceFeature.getImage().getImageID()}', ref)
            img = feature.getImage().computeFeatureMap() 
            epiline_x = (int(-epiline[2] / epiline[0]), 0)
            epiline_y = (int((-epiline[2] - (epiline[1]*480)) / epiline[0]), 480)
            cv.line(img, epiline_x, epiline_y, (255, 0, 0), 1)
            cv.circle(img, (int(feature.getX()), int(feature.getY())), 3, (0, 255, 0), -1)
            cv.imshow(f'Sensed Image ID : {feature.getImage().getImageID()}', img)
            cv.waitKey(0)
            cv.destroyAllWindows()

    return potentialFeatures

def sortPotentialFeatures(feature, potentialFeatures, referenceImage) : 
    for potentialFeature in potentialFeatures : 
        sensedImage       = potentialFeature.getImage()
        projectionMatrix1 = referenceImage.getProjectionMatrix()
        projectionMatrix2 = sensedImage.getProjectionMatrix()
        opticalCentre1    = referenceImage.getOpticalCentre() 
        opticalCentre2    = sensedImage.getOpticalCentre()
        triangulatedPoint = triangulate(feature, potentialFeature, projectionMatrix1, projectionMatrix2)
        vector1           = triangulatedPoint - opticalCentre1
        vector2           = triangulatedPoint - opticalCentre2
        # depth             = abs(norm((vector1)) - norm((vector2)))
        depth             = norm(vector1)
        potentialFeature.setDepth(depth)
    potentialFeatures = insertionSortByDepth(potentialFeatures)
    logging.info(f'IMAGE {referenceImage.getImageID():02d}:Total number of potential features of {feature} : {len(potentialFeatures)}.')

    return potentialFeatures              
    
# Compute Fundamental Matrix : Multiplication of skewform epipole, projection matrix 2 and pseudoinverse of projection matrix 1 
def computeFundamentalMatrix(im1, im2) :
    logging.info(f'IMAGE {im1.getImageID():02d}:Computing Fundamental Matrix with IMAGE {im2.getImageID():02d}')
    opticalCentre1    = im1.getOpticalCentre()
    projectionMatrix1 = im1.getProjectionMatrix()
    projectionMatrix2 = im2.getProjectionMatrix()
    epipole           = projectionMatrix2 @ opticalCentre1
    epipole           = np.array([
        [ 0,         -epipole[2], epipole[1]],
        [ epipole[2], 0,         -epipole[0]],
        [-epipole[1], epipole[0], 0]
    ])
    fundamentalMatrix = epipole @ projectionMatrix2 @ pinv(projectionMatrix1)
    fundamentalMatrix = fundamentalMatrix / fundamentalMatrix[-1, -1]
    logging.debug(f'Fundamental Matrix : {fundamentalMatrix}')

    return fundamentalMatrix

def insertionSortByDepth(A) : 
    i = 1 
    while i < len(A) : 
        j = i 
        while j > 0 and A[j-1].getDepth() > A[j].getDepth() : 
            temp   = A[j] 
            A[j]   = A[j-1]
            A[j-1] = temp
            j = j - 1 
        i = i + 1 
    
    return A

def triangulate(f1, f2, m1, m2) : 
    u1 = f1.getX() 
    v1 = f1.getY()
    u2 = f2.getX()
    v2 = f2.getY()

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

def computeNCC(gridVal1, gridVal2) : 
    length = 25
    mean1 = 0 
    mean2 = 0
    for i in range(gridVal1.shape[0]) :
        for j in range(gridVal1.shape[0]) : 
            mean1 += gridVal1[i][j]
            mean2 += gridVal2[i][j]
    mean1 /= length
    mean2 /= length
    product = 0
    std1 = 0
    std2 = 0
    for i in range(gridVal1.shape[0]) :
        for j in range(gridVal1.shape[0]) : 
            diff1 = gridVal1[i][j] - mean1 
            diff2 = gridVal2[i][j] = mean2
            product += diff1 * diff2
            std1 += diff1**2
            std2 += diff2**2
    stds = std1* std2
    if stds == 0 :
        return 0 
    else :
        return product / sqrt(stds)

def bilinearInterpolationModule(img, grid) : 
    gridVal = np.empty((5, 5))
    gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for i in range(grid.shape[0]) :
        for j in range(grid.shape[1]) :
            x   = grid[i][j][0]
            y   = grid[i][j][1]
            x1  = int(grid[i][j][0]) 
            x2  = int(grid[i][j][0]) + 1
            y1  = int(grid[i][j][1]) 
            y2  = int(grid[i][j][1]) + 1
            q11 = gray1[x1][y1]
            q12 = gray1[x1][y2]
            q21 = gray1[x2][y1]
            q22 = gray1[x2][y2]
            gridVal[i][j] = computeBilinearInterpolation(x, y, x1, x2, y1, y2, q11, q12, q21, q22)

    return gridVal

def computeBilinearInterpolation(x, y, x1, x2, y1, y2, q11, q12, q21, q22) : 
    a = 1 / ((x2 - x1) * (y2 - y1))
    b = np.array([x2-x, x-x1])
    c = np.array([
        [q11, q12],
        [q21, q22]
    ])
    d = np.array([
        [y2 - y],
        [y - y1]
    ])
    f = a * b @ c @ d

    return f 
