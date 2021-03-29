class Feature : 
    # Default Constructor 
    def __init__(self, x, y, image) : 
        self.x     = x 
        self.y     = y
        self.image = image
        self.depth = None
    # Setters
    def setX(self, x) : 
        self.x = x 
    def setY(self, y) : 
        self.y = y 
    def setImage(self, image) : 
        self.image = image
    def setDepth(self, depth) : 
        self.depth = depth
    # Getters 
    def getX(self) : 
        return self.x 
    def getY(self) : 
        return self.y
    def getImage(self) : 
        return self.image
    def getDepth(self) : 
        return self.depth