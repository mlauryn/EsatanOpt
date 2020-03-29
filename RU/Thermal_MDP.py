import openmdao.api as om
import numpy as np

from Solar import Solar
from GMM_group import GMM
from Thermal_Cycle import Thermal_Cycle
from PowerOutput import PowerOutput
from PowerInput import PowerInput

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
        n_in = len(self.opticals)

        self.add_subsystem('gmm', GMM(n=nn, conductors=self.conductors, opticals=self.opticals, 
                                area=self.area, SF = self.SF, VF=self.VF, GL_init=self.GL_init, GR_init=GR_init), promotes=['*'])
        self.add_subsystem('sol', Solar(npts=npts, area=self.area), promotes=['*'])
        self.add_subsystem('tc', Thermal_Cycle(nn=nn, npts=npts, nodes=self.opticals), promotes=['*'])
        self.add_subsystem('Pout', PowerOutput(nn=nn, npts=npts), promotes=['*'])
        self.add_subsystem('Pin', PowerInput(n_in=n_in, npts=npts), promotes=['*'])
        self.add_subsystem('Ptot', om.ExecComp('P_tot = P_in - P_out', P_in=np.ones(npts), P_out=np.ones(npts), P_tot = np.ones(npts)), promotes=['*'])
        self.add_subsystem('obj', om.ExecComp('obj = sum(P_out)', P_out=np.ones(npts)), promotes=['*'] )

        # constraint aggregation
        self.add_subsystem('power_bal', om.KSComp(width=npts, lower_flag=True))
        self.add_subsystem('bat_lwr', om.KSComp(width=npts, upper=273., lower_flag=True))
        self.add_subsystem('bat_upr', om.KSComp(width=npts, upper=45.+273.))
        self.add_subsystem('prop_upr', om.KSComp(width=npts, upper=80.+273.))
        self.add_subsystem('prop_lwr', om.KSComp(width=npts, upper=-10.+273., lower_flag=True))

        self.connect('P_tot', 'power_bal.g')
        self.connect('T', 'bat_lwr.g', src_indices=[[(-1,0), (-1,1)]])
        self.connect('T', 'bat_upr.g', src_indices=[[(-1,0), (-1,1)]])
        self.connect('T', 'prop_lwr.g', src_indices=[[(-4,0), (-4,1)]])
        self.connect('T', 'prop_upr.g', src_indices=[[(-4,0), (-4,1)]])

if __name__ == "__main__":

    from Conductors import parse_cond
    from ViewFactors import parse_vf
    from inits import inits
    
    
    npts = 2
    nodals = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'
    nn, GL_init, GR_init, QI_init, QS_init = inits(nodals, conductors)

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

    n = len(vf_nodes)

    params = model.add_subsystem('params', om.IndepVarComp(), promotes=['*'])
    params.add_output('QI', val=np.repeat(QI_init, npts, axis=1))
    params.add_output('beta', val=[90., 90.], units='deg' )
    params.add_output('dist', val=[1., 3.])
    params.add_output('alp_r', shape=(n,1), desc='absorbtivity of the input node radiating surface')
    params.add_output('cr', shape=(n,1), desc='solar cell or radiator installation decision for input nodes')
    for var in k:
        params.add_output(var, val=k[var]) # adds output variable and initial value with the same name as user conductor name

    for i,node in enumerate(vf_nodes):
        name = 'eps:{}'.format(node)
        params.add_output(name, val=eps[i]) # adds output variable and initial value as 'emissivity:node no.'
    
    prob = om.Problem(model=model)

    model.add_design_var('Spacer5', lower=0.02, upper=200.)
    model.add_design_var('Spacer1', lower=0.02, upper=200.)
    model.add_design_var('cr', lower=0.0, upper=1., indices=[0,5])
    model.add_design_var('alp_r', lower=0.02, upper=0.89, indices=[0,5])
    model.add_design_var('eps:6', lower=0.02, upper=0.89)
    model.add_design_var('eps:8', lower=0.02, upper=0.89)
    model.add_design_var('eps:9', lower=0.02, upper=0.89)
    model.add_design_var('eps:10', lower=0.02, upper=0.89)
    model.add_design_var('eps:11', lower=0.02, upper=0.89)
    model.add_design_var('QI', lower=0.2, upper=1.0, indices=[-1, -2, -7, -8])
    model.add_design_var('beta', lower=0., upper=90.)

    #model.add_constraint('T', lower=0.+273, upper=45.+273, indices=[-1, -2])
    model.add_constraint('power_bal.KS', upper=0.0)
    model.add_constraint('bat_lwr.KS', upper=0.0)
    model.add_constraint('bat_upr.KS', upper=0.0)
    model.add_constraint('prop_upr.KS', upper=0.0)
    model.add_constraint('prop_lwr.KS', upper=0.0)


    model.add_objective('obj')
    model.linear_solver = om.DirectSolver()
    model.linear_solver.options['assemble_jac'] = False

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer']='SLSQP'
    prob.driver.options['disp'] = True
    prob.driver.options['maxiter'] = 150
    #prob.driver.opt_settings = {'eps': 1.0e-1, 'ftol':1e-04,}
    prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']
    prob.driver.add_recorder(om.SqliteRecorder("thermal_mdp.sql"))

    prob.setup(check=True, mode='fwd')
    
    prob.run_model()
    
    #totals = prob.compute_totals(of=['T'], wrt=['Spacer5'])
    #print(totals)
    #prob.check_totals(compact_print=True)

    prob.run_driver()

    print(prob['T']-273.)
    #print(prob['eta'])
    print(prob['power_bal.KS'], prob['bat_lwr.KS'], prob['bat_upr.KS'], prob['prop_upr.KS'], prob['prop_lwr.KS'])

    #print(prob['P_tot'])
    #print(prob['QS_c'], prob['QS_r'])