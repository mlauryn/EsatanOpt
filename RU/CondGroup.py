import openmdao.api as om
import numpy as np
from GLmtxComp import GLmtxComp
from TempComp import TempComp

class CondGroup(om.Group):
    def __init__(self, n, nodes, SF, GL_init, GR_init, QS_init, QI_init):
            super(CondGroup, self).__init__()

            self.SF = SF
            self.nodes = nodes
            self.GL_init = GL_init
            self.GR_init = GR_init
            self.n = n
            self.QS_init = QS_init
            self.QI_init = QI_init
    
    def setup(self):
        
        n = self.n
        GL_init = self.GL_init
        nodes = self.nodes
        SF = self.SF
        
        param = om.IndepVarComp()
        for var in nodes:
            param.add_output(var) # adds output variable with the same name as user conductor name
        param.add_output('QS', val=self.QS_init)
        param.add_output('QI', val=self.QI_init)
        param.add_output('GR', val=self.GR_init)

        self.add_subsystem('param', param, promotes=['*'])
        self.add_subsystem('GLmtx', GLmtxComp(n=n, GL_init=GL_init, nodes=nodes, SF=SF), promotes=['*'])
        self.add_subsystem('temp', TempComp(n=n), promotes_inputs=['*'])
        self.add_subsystem('obj', om.ExecComp('obj = 0 - T[12]', T=np.ones(n), obj=1.0), promotes=['*'])

        self.temp.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.temp.nonlinear_solver.options['iprint'] = 2
        self.temp.nonlinear_solver.options['maxiter'] = 50
        #self.temp.nonlinear_solver.options['debug_print'] = True
        """ self.temp.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.temp.nonlinear_solver.linesearch.options['maxiter'] = 10
        self.temp.nonlinear_solver.linesearch.options['iprint'] = 2 """
        self.temp.linear_solver = om.DirectSolver()


    
if __name__ == "__main__":
    # test script for trivial optimization to maximize battery temperature
    from Conductors import _parse_line, parse_cond
    from inits import inits

    # define boundary conditions
    n = 13
    nodes = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'

    GL_init, GR_init, QI_init, QS_init = inits(n, nodes, conductors)
    
    GR_init = GR_init[1:,0]

    filepath = 'conductors.txt'
    data = parse_cond(filepath)
    nodes = {}
    shape_factors = {}
    values = {}
    for entry in data:
        nodes.update( {entry['cond_name'] : tuple(map(int, entry['nodes'].split(',')))} )
        shape_factors.update( {entry['cond_name'] : float(entry['SF']) } )
        values.update( {entry['cond_name'] : float(entry['conductivity']) } ) 

    model = CondGroup(n=n, nodes=nodes, SF=shape_factors, GL_init=GL_init, GR_init=GR_init, QI_init=QI_init, QS_init=QS_init)

    model.connect('temp.T', 'T')
    prob = om.Problem(model)
    model.add_design_var('Spacer5', lower=4., upper=200.0)

    #objective function is to minimize battery temp
    model.add_objective('obj')
    prob.setup(check=True)

    # initial conductor values
    for var in nodes:
        prob[var] = values[var]

    prob.run_model()

    #prob.model.list_outputs(print_arrays=True)
    #print(prob['temp.T']-273.15)
    

    prob.set_solver_print(level=1)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer']='SLSQP'
    prob.driver.options['disp'] = True
    #prob.driver.opt_settings = {'eps': 1.0e-1, 'ftol':1e-04,}
    prob.driver.options['debug_print'] = ['desvars', 'objs', 'totals']
    
    #totals = prob.compute_totals(of=['obj'], wrt=['Spacer5'])
    #print(totals)
    #prob.check_totals(compact_print=True)

    prob.run_driver()

    #prob.model.list_inputs(print_arrays=True, includes=['*GL*'])

    print(prob['temp.T']-273.15, prob['Spacer5'])

