class Patch : 
    # Default Constructor
    def __init__(self, centre, normal, ref) : 
        self.centre = centre
        self.normal = normal
        self.ref = ref 
        self.Vp = None
        self.px = None
        self.py = None
        self.gridVal = None