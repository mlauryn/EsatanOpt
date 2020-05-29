import os
import openmdao.api as om
import numpy as np
from Pre_process import parse_vf, parse_cond, inits, conductors, nodes, opticals, idx_dict
from Solar import Solar
from GMM_group import GMM
from Thermal_Cycle import Thermal_Cycle
from PowerOutput import PowerOutput
from PowerInput import PowerInput

class Thermal_MDF(om.Group):
    def __init__(self, npts, model, labels):
        super(Thermal_MDF, self).__init__()

        fpath = os.path.dirname(os.path.realpath(__file__))
        model_dir = fpath + '/Esatan_models/' + model
        nn, groups = nodes(data=model_dir+'/nodes_output.csv')
        
        self.GL_init, self.GR_init = conductors(nn=nn, data=model_dir+'/cond_output.csv')
        self.conductors = parse_cond(filepath=model_dir+'/cond_report.txt')
        optprop = parse_vf(filepath=model_dir+'/vf_report.txt')
        self.faces = opticals(groups, labels, optprop) 
        self.nn = nn
        self.npts = npts
        self.model = model

        N = []
        areas = []
        for face in self.faces:
            N.extend(face['nodes'])
            areas.extend(face['areas'])

        self.N = np.array(N)
        areas = np.array(areas)
        self.areas = areas[self.N.argsort()] # sort areas by ascending node number
        self.N.sort() # sort node numbers ascending

    def setup(self):
        nn = self.nn
        npts = self.npts
        n_in = len(self.N)

        params = self.add_subsystem('params', om.IndepVarComp(), promotes=['*'])
        params.add_output('QI', val=np.zeros((nn+1, npts)), units='W')
        params.add_output('phi', val=np.array([0.,0.]), units='deg' )
        params.add_output('dist', val=np.array([1., 3.]))
        params.add_output('alp_r', val=np.zeros(n_in), desc='absorbtivity of the input node radiating surface')
        params.add_output('cr', val=np.zeros(n_in), desc='solar cell or radiator installation decision for input nodes')
        for cond in self.conductors:
            params.add_output(cond['cond_name'], val=cond['values'][0] ) # adds output variable with the same name as user conductor name
        for face in self.faces:
            params.add_output(face['name'], val=face['eps'][0] ) # adds independant variable as face name and assigns emissivity of it's first node

        self.add_subsystem('gmm', GMM(n=nn, conductors=self.conductors, faces=self.faces, 
                            GL_init=self.GL_init, GR_init=self.GR_init), promotes=['*'])
        self.add_subsystem('sol', Solar(npts=npts, areas=self.areas, nodes=self.N, model=self.model), promotes=['*'])
        self.add_subsystem('tc', Thermal_Cycle(nn=nn, npts=npts, nodes=self.N), promotes=['*'])
        self.add_subsystem('Pout', PowerOutput(nn=nn, npts=npts), promotes=['*'])
        self.add_subsystem('Pin', PowerInput(n_in=n_in, npts=npts), promotes=['*'])
        
        # objective is minimize power consumption
        #self.add_subsystem('Pdis', om.ExecComp('P_dis = P_in - P_out', P_in=np.ones(npts), P_out=np.ones(npts), P_dis = np.ones(npts)), promotes=['*'])
        #self.add_subsystem('obj', om.ExecComp('obj = sum(P_out)', P_out=np.ones(npts)), promotes=['*'] )
        
        #objective is maximize total power input
        #self.add_subsystem('obj', om.ExecComp('obj = -sum(P_in)', P_in=np.ones(npts)), promotes=['*'] )
         
        #obj is to maximise prop power 
        self.add_subsystem('obj', om.ExecComp('obj = -sum(P_prop)', P_prop=np.ones(npts)), promotes=['*'] )
        self.connect('QI', 'P_prop', src_indices=[(-4,0), (-4,1)])

        # equality contraint for keeping Pin=Pout (conservation of energy)
        equal = om.EQConstraintComp()
        self.add_subsystem('equal', equal)
        equal.add_eq_output('power_bal', shape=npts, add_constraint=True, normalize=False, eq_units='W')
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
  
    npts = 2
    model_name = 'RU_v4_detail'
    #keys = list(groups.keys()) # import all nodes?
    keys = ['Box:outer',
        'Panel_outer:solar_cells',
        'Panel_inner:solar_cells',
        'Panel_body:solar_cells',
        #'Panel_inner: back',
        #'Panel_outer:back',
    ] # define faces to include in radiative analysis
    
    fpath = os.path.dirname(os.path.realpath(__file__))
    model_dir = fpath + '/Esatan_models/' + model_name
    data = model_dir+'/nodes_output.csv'
    nn, groups = nodes(data=data)
    nodes_list = sum([groups[group] for group in keys], [])
    #print(nodes)

    # index dictionary or radiative nodes_list
    idx = idx_dict(sorted(nodes_list), groups)

    model = Thermal_MDF(npts=npts, labels=keys, model=model_name)
    prob = om.Problem(model=model)

    model.add_design_var('Spacer5', lower=0.25, upper=237.)
    model.add_design_var('Spacer1', lower=0.25, upper=237.)
    model.add_design_var('Body_panel', lower=0.004, upper=.1)
    model.add_design_var('Hinge_middle', lower=0.02, upper=.1)
    model.add_design_var('Hinge_outer', lower=0.02, upper=.1)
    #model.add_design_var('cr', lower=0.0, upper=1., indices=list(idx['Panel_body:solar_cells'])) # only body solar cells are selected here
    model.add_design_var('alp_r', lower=0.07, upper=0.94, indices=list(idx['Box:outer'])) # optimize absorbptivity for structure
    model.add_design_var('Box:outer', lower=0.02, upper=0.94) # optimize emissivity of structure
    #model.add_design_var('Panel_outer:back', lower=0.02, upper=0.94) # optimize emissivity of solar array back surface
    #model.add_design_var('Panel_inner: back', lower=0.02, upper=0.94) # optimize emissivity of solar array back surface
    model.add_design_var('QI', lower = 0.25, upper=7., indices=[-1, -2, -7, -8, -10])
    model.add_design_var('phi', lower=0., upper=90.)

    model.add_constraint('bat_lwr.KS', upper=0.0)
    model.add_constraint('bat_upr.KS', upper=0.0)
    model.add_constraint('prop_upr.KS', upper=0.0)
    model.add_constraint('prop_lwr.KS', upper=0.0)

    model.add_objective('obj')

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer']='SLSQP'
    prob.driver.options['disp'] = True
    prob.driver.options['maxiter'] = 70
    prob.driver.options['tol'] = 1.0e-4
    #prob.driver.opt_settings['minimizer_kwargs'] = {"method": "SLSQP", "jac": True}
    #prob.driver.opt_settings['stepsize'] = 0.01
    prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']
    prob.driver.add_recorder(om.SqliteRecorder('./Cases/'+ model_name +'.sql'))

    prob.setup(check=True)

    # indices for solar cells
    sc_idx = sum([idx[keys] for keys in ['Panel_outer:solar_cells', 'Panel_inner:solar_cells', 'Panel_body:solar_cells']], [])

    # initial values for solar cells and radiators

    prob['cr'][sc_idx] = 1.0
    prob['alp_r'][list(idx['Box:outer'])] = 0.5
    prob['QI'][[-1]] = 0.2
    prob['QI'][[-4]] = 0.3
    
    # load case?
    """ cr = om.CaseReader('./Cases/RU_v4_detail_mstart_30.sql')
    cases = cr.list_cases('driver')
    num_cases = len(cases)
    print(num_cases) """

    """ # Load the last case written
    last_case = cr.get_case(cases[num_cases-1])
    best_case = cr.get_case('Opt_run3_rank0:ScipyOptimize_SLSQP|79')
    prob.load_case(best_case) """

    prob.run_model()
    #prob.run_driver()
    print(prob['T']-273.)
    #print(best_case)
    #totals = prob.compute_totals()#of=['T'], wrt=['Spacer5'])
    #print(totals)
    #check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=True, step=1e-04)
    #prob.check_totals(compact_print=True)

    #prob.model.list_inputs(print_arrays=True)
