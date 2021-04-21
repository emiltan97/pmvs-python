from numpy.core.numeric import cross
from classes import Image, Feature, Cell
import logging
import numpy as np
from numpy.linalg.linalg import inv, norm
import os
import cv2 as cv

def run(imageFile, featureFile, beta, isDisplay) : 
    print("==========================================================", flush=True)
    print("                       PREPROCESSING                      ", flush=True)
    print("==========================================================", flush=True)
    images = loadImages(imageFile)
    calibrateImages(images)
    featureDetection(images, featureFile, isDisplay)
    setCell(images, beta)

    return images

def loadImages(filename) : 
    images = [] 
    file = open(filename, 'r')
    id = 0 
    lines = file.readlines() 
    
    for line in lines : 
        words = line.split()
        name = words[0]
        ins = np.array([
            [float(words[1]), float(words[2]), float(words[3])],
            [float(words[4]), float(words[5]), float(words[6])],
            [float(words[7]), float(words[8]), float(words[9])]
        ])
        ex = np.array([
            [float(words[10]), float(words[11]), float(words[12]), float(words[19])],
            [float(words[13]), float(words[14]), float(words[15]), float(words[20])],
            [float(words[16]), float(words[17]), float(words[18]), float(words[21])]
        ])
        img = Image(name, ins, ex, id)
        images.append(img)
        id += 1 
    logging.info(f'Total Images : {len(images)}')

    return images

def calibrateImages(images) : 
    for image in images :
        logging.info(f'IMAGE {image.id:02d}:Calibrating images...')
        ins = image.ins
        ex = image.ex
        pmat = ins @ ex
        R = np.array([
            [ex[0][0], ex[0][1], ex[0][2]],
            [ex[1][0], ex[1][1], ex[1][2]],
            [ex[2][0], ex[2][1], ex[2][2]]
        ])
        t = np.array([
            ex[0][3],
            ex[1][3],
            ex[2][3]
        ])
        center = -inv(R) @ t
        center = np.array([
            center[0],
            center[1],
            center[2],
            1
        ])
        zaxis = np.array([pmat[2][0], pmat[2][1], pmat[2][2]])
        xaxis = np.array([pmat[0][0], pmat[0][1], pmat[0][2]])
        yaxis = cross(zaxis, xaxis)
        yaxis /= norm(yaxis)
        xaxis = cross(yaxis, zaxis)

        image.pmat = pmat
        image.center = center
        image.xaxis = xaxis
        image.yaxis = yaxis
        image.zaxis = zaxis

def featureDetection(images, filename, isDisplay) : 
    file = open(filename, 'r')
    lines = file.readlines()

    for image in images : 
        logging.info(f'IMAGE {image.id:02d}:Detecting features...')
        img = cv.imread(image.name)
        for line in lines : 
            words = line.split()
            if image.name == words[0] : 
                feats = []
                i = 0
                while i < int(words[1]) * 2 : 
                    feat = Feature(int(words[2+i]), int(words[3+i]), image)
                    cv.circle(img, (feat.x, feat.y), 4, (0, 0, 255), -1)
                    feats.append(feat)
                    i += 2 
        image.feats = feats
        if isDisplay : 
            # img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
            cv.imshow("test", img)
            cv.waitKey(0)
            cv.destroyAllWindows()

def setCell(images, beta) : 
    for image in images : 
        logging.info(f'IMAGE {image.id:02d}:Applying Cells')
        img = cv.imread(image.name)
        width = img.shape[0]
        height = img.shape[1]
        cells = np.empty((int(width/beta), int(height/beta)), dtype=Cell)
        y = 0 
        i = 0
        while y < height : 
            x = 0 
            j = 0
            while x < width : 
                center = np.array([x+beta/2, y+beta/2])
                cell = Cell(center) 
                cells[j][i] = cell
                j += 1
                x += beta
            i += 1
            y += beta
        image.cells = cells