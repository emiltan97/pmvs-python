import cv2 as cv
import numpy as np
import logging
from classes import Feature

def myFeats(images) : 
    file = open("features.txt", 'r')
    lines = file.readlines()

    for image in images : 
        for line in lines : 
            words = line.split()
            if image.name == words[0] : 
                feats = []
                img = cv.imread(image.name)
                i = 0
                while i < int(words[1]) * 2 : 
                    feat = Feature(int(words[2+i]), int(words[3+i]), image)
                    feats.append(feat)
                    cv.circle(img, (int(words[2+i]), int(words[3+i])), 4, (0, 0, 255), -1)
                    i += 2 
        cv.imshow(f'Image ID : {image.id}', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        image.feats = feats
        logging.info(f'IMAGE {image.id:02d}:Features detection done')
        
def SIFT(images, isDisplay) : 
    for image in images : 
        feats = []
        img = cv.imread(image.name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kps = sift.detect(gray, None) 

        for kp in kps : 
            coord = kp.pt 
            if isDisplay :
                cv.circle(img, (int(coord[0]), int(coord[1])), 4, (0, 0, 255), -1)
            feat = Feature(coord[0], coord[1], image)
            feats.append(feat)

        if isDisplay : 
            cv.imshow(f'Image {image.id}', img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
        image.feats = feats

def HarrisCorner(images, isDisplay) : 
    for image in images : 
        feats = []
        img = cv.imread(image.name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        ret, dst = cv.threshold(dst, 0.01*dst.max(), 255, 0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

        for corner in corners : 
            if isDisplay :
                cv.circle(img, (int(corner[0]), int(corner[1])), 4, (0, 0, 255), -1)
            feat = Feature(corner[0], corner[1], image)
            feats.append(feat)

        if isDisplay : 
            cv.imshow(f'Image {image.id}', img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        image.feats = feats