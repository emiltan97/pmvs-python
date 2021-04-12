from classes import Image
import numpy as np
from utils import detection

from numpy.linalg import inv

def run(filename) : 
    images = loadImages(filename)
    calibrateImages(images)
    detection.SIFT(images, True)
    # detection.HarrisCorner(images, True)

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
        id +=1 
    
    return images

def calibrateImages(images) : 
    for image in images :
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
        centre = -inv(R) @ t
        centre = np.array([
            centre[0],
            centre[1],
            centre[2],
            1
        ])
        image.pmat = pmat
        image.centre = centre