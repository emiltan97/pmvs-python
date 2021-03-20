import numpy as np
import cv2 as cv

class Image : 
    # Default Constructor
    def __init__(self, imageName, intrinsic, extrinsic, imageID, NumOfImages) : 
        self.imageName         = imageName 
        self.intrinsic         = intrinsic
        self.extrinsic         = extrinsic
        self.imageID           = imageID
        self.opticalCentre     = None 
        self.opticalAxis       = None
        self.projectionMatrix  = None 
        self.features          = None
        self.fundamentalMatrix = np.empty((NumOfImages, 3, 3))
    # Functions 
    def computeFeatureMap(self) : 
        img = cv.imread(self.imageName)
        for feature in self.features : 
            x = int(feature.getX())
            y = int(feature.getY())
            cv.circle(img, (x, y), 4, (0, 0, 255), -1)

        return img 
    # Setters
    def setImageName(self, imageName) : 
        self.imageName = imageName
    def setIntrinsic(self, intrinsic) : 
        self.intrinsic = intrinsic 
    def setExtrinsic(self, extrinsic) : 
        self.extrinsic = extrinsic 
    def setOpticalCentre(self, opticalCentre) : 
        self.opticalCentre = opticalCentre
    def setOpticalAxis(self, opticalAxis) : 
        self.opticalAxis = opticalAxis
    def setProjectionMatrix(self, projectionMatrix) : 
        self.projectionMatrix = projectionMatrix
    def setImageID(self, imageID) : 
        self.imageID = imageID
    def setFundamentalMatrix(self, targetImageID, fundamentalMatrix) : 
        self.fundamentalMatrix[targetImageID] = fundamentalMatrix
    def setFeatures(self, features) : 
        self.features = features
    # Getters
    def getImageName(self) : 
        return self.imageName
    def getIntrinsic(self) : 
        return self.intrinsic
    def getExtrinsic(self) : 
        return self.extrinsic
    def getOpticalCentre(self) : 
        return self.opticalCentre
    def getOpticalAxis(self) : 
        return self.opticalAxis
    def getProjectionMatrix(self) : 
        return self.projectionMatrix
    def getImageID(self) : 
        return self.imageID
    def getFundamentalMatrix(self, targetImageID) : 
        return self.fundamentalMatrix[targetImageID]
    def getFeatures(self) : 
        return self.features