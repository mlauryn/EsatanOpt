import openmdao.api as om
import numpy as np
from GLmtxComp import GLmtxComp
from GRmtxComp import GRmtxComp

class Cond_group(om.Group):
    def __init__(self, n, conductors, faces, GL_init, GR_init):
            super(Cond_group, self).__init__()

            self.n = n
            self.conductors = conductors
            self.faces = faces
            self.GL_init = GL_init
            self.GR_init = GR_init
    
    def setup(self):
        
        nn = self.n
        
        self.add_subsystem('GLmtx', GLmtxComp(n=nn, GL_init=self.GL_init, user_links=self.conductors), promotes=['*'])
        self.add_subsystem('GRmtx', GRmtxComp(n=nn, GR_init=self.GR_init, faces=self.faces), promotes=['*'])