from numpy import dot
import numpy as np
import cv2 as cv
from numpy.core.numeric import cross
from numpy.linalg.linalg import norm, pinv

class Image : 
    def __init__(self, name, ins, ex, id) : 
        self.name = name 
        self.id = id 
        self.ins = ins 
        self.ex = ex 
        self.pmat = None
        self.center = None
        self.feats = None
        self.xaxis = None 
        self.yaxis = None
        self.zaxis = None
        self.cells = None

    def displayFeatureMap(self) : 
        img = cv.imread(self.name)
        for feat in self.feats : 
            cv.circle(img, (feat.x, feat.y), 4, (0, 0, 255), -1)

        return img

class Feature : 
    def __init__(self, x, y, image) :
        self.x = x 
        self.y = y 
        self.image = image 
        self.depth = None 
    
class Patch : 
    def __init__(self, center, normal, ref) : 
        self.center = center 
        self.normal = normal 
        self.ref = ref 
        self.px, self.py = self.getPatchAxes()
        self.cells = []
        self.VpStar = []
        self.Vp = []

    def getPatchAxes(self) : 
        pmat= self.ref.pmat
        xaxis = self.ref.xaxis
        yaxis = self.ref.yaxis
        normal3 = np.array([self.normal[0], self.normal[1], self.normal[2]])
        yaxis3 = cross(normal3, xaxis)
        yaxis3 /= norm(yaxis3)
        xaxis3 = cross(yaxis3, normal3)
        xaxe = np.array([xaxis[0], xaxis[1], xaxis[2], 0])
        yaxe = np.array([yaxis[0], yaxis[1], yaxis[2], 0])
        fx = xaxe @ pmat[0]
        fy = yaxe @ pmat[1]
        fz = norm(self.center - self.ref.center)
        ftmp = fx + fy
        pscale = fz / ftmp
        pscale = 2 * fz / ftmp
        pxaxis = np.array([xaxis3[0], xaxis3[1], xaxis3[2], 0])
        pyaxis = np.array([yaxis3[0], yaxis3[1], yaxis3[2], 0])

        pxaxis *= pscale
        pyaxis *= pscale

        a = pmat @ (self.center + pxaxis)
        b = pmat @ (self.center + pyaxis)
        c = pmat @ (self.center)
        a /= a[2]
        b /= b[2]
        c /= c[2]

        xdis = norm(a - c)
        ydis = norm(b - c)

        if xdis != 0 :
            pxaxis /= xdis 
        if ydis != 0 :
            pyaxis /= ydis

        return pxaxis, pyaxis

class Cell : 
    def __init__(self, center, image) :
        self.q = []
        self.qStar = []
        self.center = center
        self.feats = []
        self.hasRecon = False
        self.image = image