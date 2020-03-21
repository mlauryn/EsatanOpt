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
        self.add_subsystem('Temp', TempComp(n=n), promotes=['*'])

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 40
        """ self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.linesearch.options['maxiter'] = 10
        self.nonlinear_solver.linesearch.options['iprint'] = 2 """
        self.linear_solver = om.DirectSolver()

if __name__ == "__main__":
    

    from Conductors import _parse_line, parse_cond
    from inits import inits

    # define boundary conditions
    n = 13
    env = '99999'
    inact = '99998'
    nodes = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'
    GL_init, GR_init, QI_init, QS_init = inits(n, env, inact, nodes, conductors)

    filepath = 'conductors.txt'
    data = parse_cond(filepath)
    nodes = {}
    shape_factors = {}
    values = {}
    for entry in data:
        nodes.update( {entry['cond_name'] : tuple(map(int, entry['nodes'].split(',')))} )
        shape_factors.update( {entry['cond_name'] : float(entry['SF']) } )
        values.update( {entry['cond_name'] : float(entry['conductivity']) } ) 

    #esatan does not include stefan-boltzman const
    sigma = 5.670374e-8
    GR_init = GR_init[1:,0]*sigma    

    model = CondGroup(n=n, nodes=nodes, SF=shape_factors, GL_init=GL_init, GR_init=GR_init, QI_init=QI_init, QS_init=QS_init)
    p = om.Problem(model)
    p.setup(check=True)

    # initial conductor values
    for var in nodes:
        p[var] = values[var]
   

    #gues values
    p['T'] = -np.ones(n)*60 + 273
    p.run_model()

    #p.model.list_outputs(print_arrays=True)

    print(p['T']-273.15)