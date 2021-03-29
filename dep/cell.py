class Cell : 
    # Default Constructor
    def __init__(self, topLeft, content, index) : 
        self.topLeft = topLeft
        self.content = content
        self.index = index
        self.patch = None 
    # Setters 
    def setTopLeft(self, topLeft) : 
        self.topLeft = topLeft
    def setPatch(self, patch) : 
        self.patch = patch
    def setContent(self, content) : 
        self.content = content
    # Getters 
    def getTopLeft(self) : 
        return self.topLeft
    def getPatch(self) : 
        return self.patch
    def getContent(self) : 
        return self.content