class Patch : 
    # Default Constructor
    def __init__(self, centre, normal, referenceImage) : 
        self.centre                 = centre 
        self.normal                 = normal 
        self.referenceImage         = referenceImage
        # self.topLeft                = topLeft
        # self.bottomRight            = bottomRight
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
    def setTopLeft(self, topLeft) : 
        self.topLeft = topLeft
    def setBottomRight(self, bottomRight) : 
        self.bottomRight = bottomRight
    # Getters 
    def getCentre(self) : 
        return self.centre
    def getNormal(self) : 
        return self.normal
    def getReferenceImage(self) : 
        return self.referenceImage
    def getPotentialVisibleImages(self) : 
        return self.potentialVisibleImages
    def getTopLeft(self) : 
        return self.topLeft
    def getBottomRight(self) : 
        return self.bottomRight