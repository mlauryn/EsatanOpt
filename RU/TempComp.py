import openmdao.api as om
import numpy as np

class TempComp(om.ImplicitComponent):
    """Computes steady state node temperature residual across a model based on conductor definition and boundary conditions at single design point."""
    def initialize(self):
        self.options.declare('n', default=1, types=int, desc='number of diffusion nodes')
    def setup(self):
        n = self.options['n'] + 1
        self.add_output('T', val=np.zeros(n), units='K')
        self.add_input('GL', val=np.zeros((n,n)), units='W/K')
        self.add_input('GR', val=np.zeros((n,n)))
        self.add_input('QS', val=np.zeros(n), units='W')
        self.add_input('QI', val=np.zeros(n), units='W')
        
        rows = np.arange(n).repeat(n)
        cols = np.arange(n**2)
        
        self.declare_partials(of='T', wrt='G*', cols=cols, rows=rows)
        self.declare_partials(of='T', wrt='Q*', cols=np.arange(n), rows=np.arange(n), val=1.0)
        self.declare_partials(of='T', wrt='T')

    def apply_nonlinear(self, inputs, outputs, residuals):
        GL = inputs['GL']
        GR = inputs['GR']
        QS = inputs['QS']
        QI = inputs['QI']
        T = outputs['T']

        residuals['T'] = GL.dot(T) + GR.dot(T**4) + QS + QI
    def linearize(self, 
    inputs, outputs, partials):
        n = self.options['n'] + 1
        GL = inputs['GL']
        GR = inputs['GR']
        T = outputs['T']

        partials['T', 'GL'] = np.resize(T, n*n)
        partials['T', 'GR'] = np.resize(T**4, n*n)
        partials['T', 'T'] = (GL + (4 * (GR * (T ** 3))))
    
    def guess_nonlinear(self, inputs, outputs, residuals):
        n = self.options['n'] + 1
        #gues values
        outputs['T'] = -np.ones(n)*20 + 273

if __name__ == "__main__":
    from Pre_process import nodes, conductors, inits
    
    model_name = 'RU_v5_1'
    nn, groups, output = nodes(data='./Esatan_models/'+model_name+'/nodes_output.csv')
    GL_init, GR_init = conductors(nn=nn, data='./Esatan_models/'+model_name+'/cond_output.csv')
    QI_init, QS_init = inits(data='./Esatan_models/'+model_name+'/nodes_output.csv')
    
    problem = om.Problem()
    model = problem.model

    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('GL', val=GL_init, units='W/K')
    indeps.add_output('GR', val=GR_init)
    indeps.add_output('QS', val=QS_init, units='W')
    indeps.add_output('QI', val=QI_init, units='W')

    model.add_subsystem('tmm', TempComp(n=nn), promotes=['*'])

    model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=True
        )
    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 50
    """ model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
    model.nonlinear_solver.linesearch.options['maxiter'] = 10
    model.nonlinear_solver.linesearch.options['iprint'] = 2 """
    model.linear_solver = om.DirectSolver(assemble_jac=True)
    #model.options['assembled_jac_type'] = 'csc'

    problem.setup(check=True)

    problem.run_model()
    print(problem['T']-273.15)

    #check_partials_data = problem.check_partials(compact_print=False, show_only_incorrect=True, form='central', step=1e-3)
    #problem.model.list_inputs(print_arrays=True, includes=['*G*'])