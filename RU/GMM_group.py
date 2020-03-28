import openmdao.api as om
import numpy as np
from GLmtxComp import GLmtxComp
from GRmtxComp import GRmtxComp

class GMM(om.Group):
    def __init__(self, n, conductors, opticals, area, SF, VF, GL_init, GR_init):
            super(GMM, self).__init__()

            self.n = n
            self.SF = SF
            self.VF = VF
            self.conductors = conductors
            self.opticals = opticals
            self.area = area
            self.GL_init = GL_init
            self.GR_init = GR_init
    
    def setup(self):
        
        n = self.n
        
        self.add_subsystem('GLmtx', GLmtxComp(n=n, GL_init=self.GL_init, nodes=self.conductors, SF=self.SF), promotes=['*'])
        self.add_subsystem('GRmtx', GRmtxComp(n=n, GR_init=self.GR_init, nodes=self.opticals, VF=self.VF, A=self.area), promotes=['*'])