import openmdao.api as om
import numpy as np

from Solar import Solar
from GMM_group import GMM
from Thermal_Cycle import Thermal_Cycle
from PowerOutput import PowerOutput
from PowerInput import PowerInput

class RU_MDP(om.Group):
    def __init__(self, nn, npts, opticals, area, conductors, SF, VF, GL_init, GR_init):
        super(RU_MDP, self).__init__()

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
        
        # objective is minimize power consumption
        #self.add_subsystem('Pdis', om.ExecComp('P_dis = P_in - P_out', P_in=np.ones(npts), P_out=np.ones(npts), P_dis = np.ones(npts)), promotes=['*'])
        self.add_subsystem('obj', om.ExecComp('obj = sum(P_out)', P_out=np.ones(npts)), promotes=['*'] )
        
        #objective is maximize total power input
        #self.add_subsystem('obj', om.ExecComp('obj = -sum(P_in)', P_in=np.ones(npts)), promotes=['*'] )
         
        #obj is to maximise prop power 
        """ self.add_subsystem('obj', om.ExecComp('obj = -sum(P_prop)', P_prop=np.ones(npts)), promotes=['*'] )
        self.connect('QI', 'P_prop', src_indices=[(-4,0), (-4,1)]) """

        # equality contraint for keeping Pin=Pout (conservation of energy)
        equal = om.EQConstraintComp()
        self.add_subsystem('equal', equal)
        equal.add_eq_output('power_bal', shape=npts, add_constraint=True, normalize=False)
        self.connect('P_out', 'equal.lhs:power_bal')
        self.connect('P_in', 'equal.rhs:power_bal')

        # temperature constraint aggregation
        #self.add_subsystem('power_bal', om.KSComp(width=npts, lower_flag=True))
        self.add_subsystem('bat_lwr', om.KSComp(width=npts, upper=273., lower_flag=True))
        self.add_subsystem('bat_upr', om.KSComp(width=npts, upper=45.+273.))
        self.add_subsystem('prop_upr', om.KSComp(width=npts, upper=80.+273.))
        self.add_subsystem('prop_lwr', om.KSComp(width=npts, upper=-10.+273., lower_flag=True))

        # min power consumption
        """ self.add_subsystem('obc_pwr', om.KSComp(width=npts, upper=0.2, lower_flag=True))
        self.add_subsystem('prop_pwr', om.KSComp(width=npts, upper=0.3, lower_flag=True)) """

        #self.connect('P_dis', 'power_bal.g')
        self.connect('T', 'bat_lwr.g', src_indices=[[(-1,0), (-1,1)]])
        self.connect('T', 'bat_upr.g', src_indices=[[(-1,0), (-1,1)]])
        self.connect('T', 'prop_lwr.g', src_indices=[[(-4,0), (-4,1)]])
        self.connect('T', 'prop_upr.g', src_indices=[[(-4,0), (-4,1)]])
        """ self.connect('QI', 'obc_pwr.g', src_indices=[[(-1,0), (-1,1)]])
        self.connect('QI', 'prop_pwr.g', src_indices=[[(-4,0), (-4,1)]]) """

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

    model = RU_MDP(nn=nn, npts=npts, opticals=vf_nodes, area=area, conductors=cond_nodes, SF=shape_factors, VF=VF, GL_init=GL_init, GR_init=GR_init)

    n = len(vf_nodes)

    params = model.add_subsystem('params', om.IndepVarComp(), promotes=['*'])
    params.add_output('QI', val=np.repeat(QI_init, npts, axis=1))
    params.add_output('beta', val=np.asarray([60., 30.]), units='deg' )
    params.add_output('dist', val=[1., 3.])
    params.add_output('alp_r', shape=(n,1), desc='absorbtivity of the input node radiating surface')
    params.add_output('cr', shape=(n,1), desc='solar cell or radiator installation decision for input nodes')
    for var in k:
        params.add_output(var, val=k[var]) # adds output variable and initial value with the same name as user conductor name

    for i,node in enumerate(vf_nodes):
        name = 'eps:{}'.format(node)
        params.add_output(name, val=eps[i]) # adds output variable and initial value as 'emissivity:node no.'
    
    prob = om.Problem(model=model)

    model.add_design_var('Spacer5', lower=0.25, upper=237.)
    model.add_design_var('Spacer1', lower=0.25, upper=237.)
    model.add_design_var('Hinge_inner', lower=0.004, upper=.1)
    model.add_design_var('Hinge_middle', lower=0.02, upper=.1)
    model.add_design_var('Hinge_outer', lower=0.02, upper=.1)

    model.add_design_var('cr', lower=0.0, upper=1., indices=[0,5])
    model.add_design_var('alp_r', lower=0.07, upper=0.94, indices=[0,5])
    model.add_design_var('eps:2', lower=0.02, upper=0.94)
    model.add_design_var('eps:5', lower=0.02, upper=0.94)
    model.add_design_var('eps:6', lower=0.02, upper=0.94)
    model.add_design_var('eps:8', lower=0.02, upper=0.94)
    model.add_design_var('eps:9', lower=0.02, upper=0.94)
    model.add_design_var('eps:10', lower=0.02, upper=0.94)
    model.add_design_var('eps:11', lower=0.02, upper=0.94)
    model.add_design_var('QI', lower = 0.25, upper=7., indices=[-1, -2, -7, -8, 10])
    model.add_design_var('beta', lower=0., upper=90.)

    #model.add_constraint('T', lower=0.+273, upper=45.+273, indices=[-1, -2])
    #model.add_constraint('power_bal.KS', upper=0.0)
    model.add_constraint('bat_lwr.KS', upper=0.0)
    model.add_constraint('bat_upr.KS', upper=0.0)
    model.add_constraint('prop_upr.KS', upper=0.0)
    model.add_constraint('prop_lwr.KS', upper=0.0)
    """ model.add_constraint('obc_pwr.KS', upper=0.0)
    model.add_constraint('prop_pwr.KS', upper=0.0) """


    model.add_objective('obj')
    model.linear_solver = om.DirectSolver()
    model.linear_solver.options['assemble_jac'] = False

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer']='basinhopping'
    prob.driver.options['disp'] = True
    prob.driver.options['maxiter'] = 2
    prob.driver.options['tol'] = 1.0e-4
    #prob.driver.opt_settings['minimizer_kwargs'] = {"method": "SLSQP", "jac": True}
    #prob.driver.opt_settings['stepsize'] = 0.01
    prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']
    prob.driver.add_recorder(om.SqliteRecorder("ru_mdp.sql"))

    prob.setup(check=True, mode='fwd')
    
    """ cr = om.CaseReader('thermal_mdp.sql')
    cases = cr.list_cases('driver')
    num_cases = len(cases)
    print(num_cases)

    # Load the last case written
    last_case = cr.get_case(cases[num_cases-1])
    prob.load_case(last_case) """

    #last_case.list_inputs(print_arrays=True)

    prob.run_model()
    
    #totals = prob.compute_totals(of=['T'], wrt=['Spacer5'])
    #print(totals)
    #prob.check_totals(compact_print=True)

    prob.run_driver()

    print(prob['T']-273.)
    #print(prob['eta'])
    #print(prob['bat_lwr.KS'], prob['bat_upr.KS'], prob['prop_upr.KS'], prob['prop_lwr.KS']) #prob['power_bal.KS']

    #print(prob['P_dis'])
    #print(prob['P_in'], prob['P_out'])
    #print(prob['QS_c'], prob['QS_r'])

    #prob.list_inputs(print_arrays=True)