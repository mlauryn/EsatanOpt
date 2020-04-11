import openmdao.api as om
import numpy as np
from GLmtxComp import GLmtxComp
from GRmtxComp import GRmtxComp
from TempComp import TempComp

class Thermal_model(om.Group):
    def __init__(self, n, conductors, opticals, area, SF, VF, GL_init, GR_init, QS_init, QI_init):
            super(Thermal_model, self).__init__()

            self.n = n
            self.SF = SF
            self.VF = VF
            self.conductors = conductors
            self.opticals = opticals
            self.area = area
            self.GL_init = GL_init
            self.GR_init = GR_init
            self.QS_init = QS_init
            self.QI_init = QI_init
    
    def setup(self):
        
        n = self.n
        area = self.area
        
        params = om.IndepVarComp()

        for var in self.conductors:
            params.add_output(var) # adds output variable with the same name as user conductor name

        for i,node in enumerate(self.opticals):
            name = 'eps:{}'.format(node)
            params.add_output(name) # adds output variable as 'emissivity:node no.'

        params.add_output('QS', val=self.QS_init)
        params.add_output('QI', val=self.QI_init)

        self.add_subsystem('inputs', params, promotes=['*'])
        self.add_subsystem('GLmtx', GLmtxComp(n=n, GL_init=self.GL_init, nodes=self.conductors, SF=self.SF), promotes=['*'])
        self.add_subsystem('GRmtx', GRmtxComp(n=n, GR_init=self.GR_init, nodes=self.opticals, VF=self.VF, A=self.area), promotes=['*'])
        self.add_subsystem('temp', TempComp(n=n), promotes_inputs=['*'])
        self.add_subsystem('obj', om.ExecComp('obj = 0 - T[15]', T=np.ones(n+1), obj=1.0), promotes=['*']) #trivial objective of maximising battery temperature

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        """ self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 50 """
        #self.nonlinear_solver.options['debug_print'] = True
        """ self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.linesearch.options['maxiter'] = 10
        self.nonlinear_solver.linesearch.options['iprint'] = 2 """
        self.linear_solver = om.DirectSolver()
    
if __name__ == "__main__":

    # test script for trivial optimization to maximize battery temperature

    from Conductors import parse_cond
    from ViewFactors import parse_vf
    from inits import inits

    # define boundary conditions
    
    nodals = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'

    n, GL_init, GR_init, QI_init, QS_init = inits(nodals, conductors)

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

    model = Thermal_model(n=n, conductors=cond_nodes, opticals=vf_nodes, area=area, SF=shape_factors, VF=VF, GL_init=GL_init, GR_init=GR_init, QI_init=QI_init, QS_init=QS_init)

    model.connect('temp.T', 'T')
    prob = om.Problem(model)
    model.add_design_var('Spacer5', lower=4., upper=200.0)
    model.add_design_var('eps:2', lower=.02, upper=.80)

    #objective function is to minimize battery temp
    model.add_objective('obj')

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer']='SLSQP'
    prob.driver.options['disp'] = True
    #prob.driver.opt_settings = {'eps': 1.0e-1, 'ftol':1e-04,}
    prob.driver.options['debug_print'] = ['desvars', 'objs', 'totals']

    prob.setup(check=True)

    # initial conductivity values
    for var in k:
        prob[var] = k[var]
    
    # initial emissivity values
    for i,node in enumerate(vf_nodes):
        name = 'eps:{}'.format(node)
        prob[name] = eps[i]

    prob.run_model()

    #prob.model.list_outputs(print_arrays=True)
    
    

    #prob.set_solver_print(level=1)
    
    #totals = prob.compute_totals(of=['obj'], wrt=['Spacer5'])
    #print(totals)
    prob.check_totals(compact_print=True)

    #prob.run_driver()

    #prob.model.list_inputs(print_arrays=True, includes=['*GR*'])

    #print(prob['temp.T']-273.15)

