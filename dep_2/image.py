import cv2 as cv

class Image : 
    # Default Constructor
    def __init__(self, name, intrinsic, extrinsic, id) : 
        self.name = name
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.id = id
        self.projectionMatrix = None
        self.opticalCentre = None 
        self.features = None
        self.grid = None
    def computeFeatureMap(self) : 
        img = cv.imread(self.name)
        for feature in self.features : 
            x = int(feature.x)
            y = int(feature.y)
            cv.circle(img, (x, y), 4, (0, 0, 255), -1)
        
        return img