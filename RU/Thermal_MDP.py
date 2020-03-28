import openmdao.api as om
import numpy as np

from Solar import Solar
from GMM_group import GMM
from Thermal_Cycle import Thermal_Cycle

class Thermal_MDP(om.Group):
    def __init__(self, nn, npts, opticals, area, conductors, SF, VF, GL_init, GR_init):
        super(Thermal_MDP, self).__init__()

        self.nn = nn
        self.npts = npts
        self.SF = SF
        self.VF = VF
        self.conductors = conductors
        self.opticals = opticals
        self.area = area
        self.GL_init = GL_init
        self.GR_init = GR_init

    def setup(self):
        nn = self.nn
        npts = self.npts

        self.add_subsystem('gmm', GMM(n=nn, conductors=self.conductors, opticals=self.opticals, 
                                area=self.area, SF = self.SF, VF=self.VF, GL_init=self.GL_init, GR_init=GR_init), promotes=['*'])
        self.add_subsystem('sol', Solar(npts=npts, area=self.area), promotes=['*'])
        self.add_subsystem('tc', Thermal_Cycle(nn=nn, npts=npts, nodes=self.opticals), promotes=['*'])

if __name__ == "__main__":

    from Conductors import parse_cond
    from ViewFactors import parse_vf
    from inits import inits
    
    npts = 2
    nodals = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'
    nn, GL_init, GR_init, QI_init, QS_init = inits(nodals, conductors)

    QI_init = QI_init[np.newaxis,:].T


    user_conductors = 'conductors.txt'
    cond_data = parse_cond(user_conductors)
    cond_nodes = {}
    shape_factors = {}
    k = {}
    for entry in cond_data:
        cond_nodes.update( {entry['cond_name'] : entry['nodes']} )
        shape_factors.update( {entry['cond_name'] : entry['SF'] } )
        k.update( {entry['cond_name'] : entry['conductivity'] } ) 

    view_factors = 'viewfactors.txt'
    vf_data = parse_vf(view_factors)
    vf_nodes = []
    area = []
    VF = []
    eps = []
    for entry in vf_data:
        vf_nodes.append(entry['node number'])
        area.append(entry['area'])
        VF.append(entry['vf']) 
        eps.append(entry['emissivity'])

    model = Thermal_MDP(nn=nn, npts=npts, opticals=vf_nodes, area=area, conductors=cond_nodes, SF=shape_factors, VF=VF, GL_init=GL_init, GR_init=GR_init)

    params = model.add_subsystem('params', om.IndepVarComp(), promotes=['*'])
    params.add_output('QI', val=np.repeat(QI_init, npts, axis=1))
    params.add_output('beta', val=np.zeros(npts) )
    params.add_output('dist', val=[1., 3.])
    for var in k:
        params.add_output(var, val=k[var]) # adds output variable and initial value with the same name as user conductor name

    for i,node in enumerate(vf_nodes):
        name = 'eps:{}'.format(node)
        params.add_output(name, val=eps[i]) # adds output variable and initial value as 'emissivity:node no.'
    
    problem = om.Problem(model=model)
    problem.setup(check=True)
    
    problem.run_model()

    print(problem['T']-273.)

