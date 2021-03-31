import numpy as np
import logging
import cv2 as cv


from PIL import Image
from image import Image
from feature import Feature
from numpy.linalg import inv
import os

def run(filename, gridSize, isDisplay) : 
    # Load the input images 
    images = loadImages(filename)
    # Calibrate each image
    calibrateImages(images)
    # Feature detection on each image
    SIFT(images, isDisplay)

    return images

def loadImages(filename) : 
    imageFile = open(filename, "r")
    images = []
    id = 0
    lines = imageFile.readlines()
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
        img = Image(imageName, intrinsic, extrinsic, id)
        images.append(img)
        id += 1
    imageFile.close()
    logging.info(f'Total Images : {len(images)}')

    return images

def calibrateImages(images) : 
    for image in images : 
        logging.info(f'IMAGE {image.id:02d}:Calibrating Image')
        intrinsic = image.intrinsic
        extrinsic = image.extrinsic
        # Compute Projection Matrix : P = K[R t]
        projectionMatrix = intrinsic @ extrinsic
        logging.debug(f'ProjectionMatrix : {projectionMatrix}')
        # Compute Optical Centre : C = -R^{-1} * t
        R = np.array([
            [extrinsic[0][0], extrinsic[0][1], extrinsic[0][2]],
            [extrinsic[1][0], extrinsic[1][1], extrinsic[1][2]],
            [extrinsic[2][0], extrinsic[2][1], extrinsic[2][2]]
        ])
        t = np.array([
            extrinsic[0][3],
            extrinsic[1][3],
            extrinsic[2][3]
        ])
        opticalCentre = -inv(R) @ t
        opticalCentre = np.array([
            opticalCentre[0],
            opticalCentre[1],
            opticalCentre[2],
            1
        ])
        logging.debug(f'Optical Centre : {opticalCentre}')
        image.projectionMatrix = projectionMatrix
        image.opticalCentre = opticalCentre

def SIFT(images, isDisplay) :
    for image in images : 
        logging.info(f'IMAGE {image.id:02d}:Applying SIFT Feature Detection')
        imageName = image.name
        img = cv.imread(imageName)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        features = []
        keypoints = sift.detect(gray, None)
        for keypoint in keypoints : 
            coordinate = keypoint.pt
            if isDisplay : 
                cv.circle(img, (int(coordinate[0]), int(coordinate[1])), 4, (0, 0, 255), -1)
            feature = Feature(coordinate[0], coordinate[1], image)
            features.append(feature) 
        if isDisplay : 
            # drawGrid(img, 32)
            cv.imshow(f'Image {image.id}', img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        image.features = features