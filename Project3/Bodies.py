class Bodies:
    '''Class that create bodies'''
    def __init__(self,name,mass,radius,pos,vc):# Convention to use self 
        ''' Self draws information from input variables to class variables.'''
        ''' Runs every time a new employee are added'''
        self.name     = name                   # Keep similar name if possible
        self.mass     = mass
        self.radius   = radius
        self.pos      = pos
        self.vc       = vc