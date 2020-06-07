import os
from openmdao.api import Group, IndepVarComp
import numpy as np
from MS import MainSP
from RU import RemoteUnit
from Pre_process import nodes, inits, idx_dict



class MAT_MDP_Group(Group):
    """ 
    Multi-asteroid touring spacecraft. Consists of main s/c and remote unit
    
    Parameters
    Sun_dist: numpy array
        vector of sun distances for orbit evaluation points
    MS_model : str
        Main s/c model name
    RU_model : str
        Remote unit model name: 

    """
    def __init__(self, Sun_dist, MS_model, MS_geom, RU_model, RU_geom):
        super(MAT_MDP_Group, self).__init__()

        self.npts = len(Sun_dist) # number of design points
        self.dist = Sun_dist
        self.MS_model = MS_model
        self.MS_geom = MS_geom
        self.RU_model = RU_model
        self.RU_geom = RU_geom

    def setup(self):

        npts = self.npts
        """ fpath = os.path.dirname(os.path.realpath(__file__))
        model_dir = fpath + '/Esatan_models/' + model_name
        data = model_dir+'/nodes_output.csv' """

        # Create IndepVarComp for broadcast parameters.
        bp = self.add_subsystem('bp', IndepVarComp())
        bp.add_output('dist', self.dist)
        bp.add_output('spinAngle', np.zeros(npts), units='deg')

        #Spacecraft models go into root group
        MAT = self.add_subsystem('MAT', Group(), promotes=['*'])

        MAT.add_subsystem('MS', MainSP(npts=npts, labels=self.MS_geom, model=self.MS_model))
        MAT.add_subsystem('RU', RemoteUnit(npts=npts, labels=self.RU_geom, model=self.RU_model))

        # Hook up broadcast inputs
        self.connect('bp.dist', 'MS.dist')
        self.connect('bp.dist', 'RU.dist')
        self.connect('bp.spinAngle', 'MS.phi')
        self.connect('bp.spinAngle', 'RU.phi')