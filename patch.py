class Patch : 
    # Default Constructor
    def __init__(self, centre, normal, xAxis, yAxis, referenceImage) : 
        self.centre                 = centre 
        self.normal                 = normal 
        self.referenceImage         = referenceImage
        self.xAxis                  = xAxis
        self.yAxis                  = yAxis
        self.potentialVisibleImages = []
    # Setters
    def setCentre(self, centre) : 
        self.centre = centre 
    def setNormal(self, normal) : 
        self.normal = normal
    def setReferenceImage(self, referenceImage) : 
        self.referenceImage = referenceImage
    def setPotentialVisibleImages(self, potentialVisibleImages) : 
        self.potentialVisibleImages = potentialVisibleImages
    def setXAxis(self, xAxis) : 
        self.xAxis = xAxis
    def setYAxis(self, yAxis) : 
        self.yAxis = yAxis
    # Getters 
    def getCentre(self) : 
        return self.centre
    def getNormal(self) : 
        return self.normal
    def getReferenceImage(self) : 
        return self.referenceImage
    def getPotentialVisibleImages(self) : 
        return self.potentialVisibleImages
    def getXAxis(self) : 
        return self.xAxis
    def getYAxis(self) : 
        return self.yAxis