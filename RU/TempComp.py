import openmdao.api as om
import numpy as np

class TempComp(om.ImplicitComponent):
    """Computes steady state node temperature residual across a model based on conductor definition and boundary conditions."""
    def initialize(self):
        self.options.declare('n', default=1, types=int, desc='number of diffusion nodes')
    def setup(self):
        n = self.options['n'] + 1
        self.add_output('T', val=np.zeros(n))
        self.add_input('GL', val=np.zeros((n,n)))
        self.add_input('GR', val=np.zeros((n,n)))
        self.add_input('QS', val=np.zeros(n))
        self.add_input('QI', val=np.zeros(n))
        self.declare_partials(of='T', wrt='*', method='fd')
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
        QS = inputs['QS']
        QI = inputs['QI']
        T = outputs['T']

        partials['T', 'GL'] = np.einsum('ij, k', np.eye(n, n), T)
        partials['T', 'GR'] = np.einsum('ij, k', np.eye(n, n), T**4)
        partials['T', 'QS'] = np.eye(n, n)
        partials['T', 'QI'] = np.eye(n, n)
        partials['T', 'T'] = (GL + (4 * (GR * (T ** 3)[np.newaxis, :])))
    
    def guess_nonlinear(self, inputs, outputs, residuals):
        n = self.options['n'] + 1
        #gues values
        outputs['T'] = -np.ones(n)*50 + 273

if __name__ == "__main__":
    from inits import inits
    problem = om.Problem()
    model = problem.model

    nodes = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'
    n, GL_init, GR_init, QI_init, QS_init = inits(nodes, conductors)

    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('GL', val=GL_init, units='W/K')
    indeps.add_output('GR', val=GR_init)
    indeps.add_output('QS', val=QS_init, units='W')
    indeps.add_output('QI', val=QI_init, units='W')

    model.add_subsystem('tmm', TempComp(n=n), promotes=['*'])

    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 50
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
    model.nonlinear_solver.linesearch.options['maxiter'] = 10
    model.nonlinear_solver.linesearch.options['iprint'] = 2
    model.linear_solver = om.DirectSolver()

    problem.setup(check=True)

    problem.run_model()
    print(problem['T']-273.15)

    check_partials_data = problem.check_partials(compact_print=True, show_only_incorrect=True, form='central', step=1e-3)
    #problem.model.list_inputs(print_arrays=True, includes=['*G*'])