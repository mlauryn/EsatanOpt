import openmdao.api as om
import numpy as np
from GLmtxComp import GLmtxComp
from GRmtxComp import GRmtxComp
from TempComp import TempComp

class ThermalGroup(om.Group):
    def __init__(self, n, user_links, faces, GL_init, GR_init, QS_init, QI_init):
            super(ThermalGroup, self).__init__()

            self.n = n
            self.user_links = user_links
            self.faces = faces
            self.GL_init = GL_init
            self.GR_init = GR_init
            self.QS_init = QS_init
            self.QI_init = QI_init
    
    def setup(self):
        
        nn = self.n
        
        params = om.IndepVarComp()

        for cond in self.user_links:
            params.add_output(cond['cond_name'], val=cond['values'][0] ) # adds output variable with the same name as user conductor name

        for face in self.faces:
            name = face['name']
            value = face['eps'][0]
            params.add_output(name, val=value ) # adds independant variable as face name and assigns emissivity of it's first node

        params.add_output('QS', val=self.QS_init)
        params.add_output('QI', val=self.QI_init)
        params.add_output('GR', val=self.GR_init)

        self.add_subsystem('inputs', params, promotes=['*'])
        self.add_subsystem('GLmtx', GLmtxComp(n=nn, GL_init=self.GL_init, user_links=self.user_links), promotes=['*'])
        #self.add_subsystem('GRmtx', GRmtxComp(n=nn, GR_init=self.GR_init, faces=self.faces), promotes=['*'])
        self.add_subsystem('temp', TempComp(n=nn), promotes_inputs=['*'])
        self.add_subsystem('obj', om.ExecComp('obj = 0 - T[73]', T=np.ones(nn+1), obj=1.0), promotes=['*']) #trivial objective of maximising battery temperature

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
    from inits import inits, nodes, conductors
    from opticals import opticals

    # define boundary conditions
    
    node_data = 'nodal_data.csv'
    cond_data = 'Cond_data.csv'
    nn, groups = nodes(node_data)
    GL_init, GR_init = conductors(nn, cond_data)
    QI_init, QS_init = inits(node_data) 

    user_links = parse_cond('conductors.txt')
    optprop = parse_vf('viewfactors.txt')
    keys = ['Box:outer', 'Panel_outer:back']
    faces = opticals(groups, keys, optprop)
    
    model = ThermalGroup(n=nn, user_links=user_links, faces=faces, GL_init=GL_init, GR_init=GR_init, QI_init=QI_init, QS_init=QS_init)

    model.connect('temp.T', 'T')
    prob = om.Problem(model)
    model.add_design_var('Spacer5', lower=4., upper=200.0)
    #model.add_design_var('eps:2', lower=.02, upper=.80)

    #objective function is to minimize battery temp
    model.add_objective('obj')
    prob.setup(check=True)

    prob.run_model()

    #prob.model.list_outputs(print_arrays=True)
    #print(prob['temp.T']-273.15)
    

    #prob.set_solver_print(level=1)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer']='SLSQP'
    prob.driver.options['disp'] = True
    #prob.driver.opt_settings = {'eps': 1.0e-1, 'ftol':1e-04,}
    prob.driver.options['debug_print'] = ['desvars', 'objs', 'totals']
    
    #totals = prob.compute_totals(of=['obj'], wrt=['Spacer5'])
    #print(totals)
    #prob.check_totals(compact_print=True)

    #prob.run_driver()

    #prob.model.list_inputs(print_arrays=True, includes=['*GR*'])

    print(prob['temp.T']-273.15)

