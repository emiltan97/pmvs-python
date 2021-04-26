from numpy.core.numeric import cross
from classes import Image, Feature, Cell
import logging
import numpy as np
from numpy.linalg.linalg import inv, norm
import cv2 as cv

def run(imageFile, featureFile, beta, isDisplay) : 
    print("==========================================================", flush=True)
    print("                       PREPROCESSING                      ", flush=True)
    print("==========================================================", flush=True)

    images = loadImages(imageFile)
    calibrateImages(images)
    setCell(images, beta, isDisplay)
    featureDetection(images, featureFile, beta, isDisplay)

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
        zaxis = np.array(pmat[2])
        zaxis[3] = 0 
        ftmp = norm(zaxis)
        zaxis /= ftmp
        zaxis = np.array([zaxis[0], zaxis[1], zaxis[2]])
        xaxis = np.array([pmat[0][0], pmat[0][1], pmat[0][2]])
        yaxis = cross(zaxis, xaxis)
        yaxis /= norm(yaxis)
        xaxis = cross(yaxis, zaxis)

        image.pmat = pmat
        image.center = center
        image.xaxis = xaxis
        image.yaxis = yaxis
        image.zaxis = zaxis

def featureDetection(images, filename, beta, isDisplay) : 
    file = open(filename, 'r')
    lines = file.readlines()

    for image in images : 
        logging.info(f'IMAGE {image.id:02d}:Detecting features...')
        img = cv.imread(image.name)
        if isDisplay : 
            x = beta 
            y = beta
            width = img.shape[1]
            height = img.shape[0]
            while x < width : 
                cv.line(img, (x, 0), (x, height), (0, 255, 0), 1)
                x += beta
            while y < height : 
                cv.line(img, (0, y), (width, y), (0, 255, 0), 1)
                y += beta
        for line in lines : 
            words = line.split()
            if image.name == words[0] : 
                feats = []
                i = 0
                while i < int(words[1]) * 2 : 
                    feat = Feature(int(words[2+i]), int(words[3+i]), image)
                    if isDisplay : 
                        cv.circle(img, (feat.x, feat.y), 4, (0, 0, 255), -1)
                    feats.append(feat)
                    a = -2 
                    while a < 3 : 
                        b = -2 
                        while b < 3 : 
                            image.cells[int(feat.y/beta+b)][int(feat.x/beta+a)].feats.append(feat)
                            if isDisplay : 
                                coord = image.cells[int(feat.y/beta+b)][int(feat.x/beta+a)].center
                                cv.circle(img, (int(coord[0]), int(coord[1])), 2, (255, 0, 0), -1)
                            b += 1
                        a += 1
                    i += 2 
        image.feats = feats
        if isDisplay : 
            cv.imshow(f'Image {image.id:02d}', img)
            cv.waitKey(0)
            cv.destroyAllWindows()

def setCell(images, beta, isDisplay) : 
    for image in images : 
        logging.info(f'IMAGE {image.id:02d}:Applying cells')
        img = cv.imread(image.name)
        width = img.shape[1]
        height = img.shape[0]
        cells = np.empty((int(height/beta), int(width/beta)), dtype=Cell)
        y = 0 
        i = 0
        while y < height : 
            x = 0 
            j = 0
            while x < width : 
                center = np.array([x+beta/2, y+beta/2])
                cell = Cell(center, image) 
                if isDisplay : 
                    cv.circle(img, (int(center[0]), int(center[1])), 2, (0, 0, 255), -1)
                cells[i][j] = cell
                j += 1
                x += beta
            i += 1
            y += beta
        image.cells = cells
        if isDisplay : 
            x = beta 
            y = beta
            while x < width : 
                cv.line(img, (x, 0), (x, height), (0, 255, 0), 1)
                x += beta
            while y < height : 
                cv.line(img, (0, y), (width, y), (0, 255, 0), 1)
                y += beta
            cv.imshow(f'Image {image.id:02d}', img)
            cv.waitKey(0)
            cv.destroyAllWindows()