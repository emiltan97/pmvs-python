class Image : 
    def __init__(self, name, ins, ex, id) : 
        self.name = name 
        self.id = id 
        self.ins = ins 
        self.ex = ex 
        self.pmat = None
        self.centre = None
        self.feats = None

class Feature : 
    def __init__(self, x, y, image) :
        self.x = x 
        self.y = y 
        self.image = image 
        self.depth = None 