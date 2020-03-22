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
        self.add_input('GR', val=np.zeros(n))
        self.add_input('QS', val=np.zeros(n))
        self.add_input('QI', val=np.zeros(n))
        self.declare_partials(of='T', wrt='*')
    def apply_nonlinear(self, inputs, outputs, residuals):
        GL = inputs['GL']
        GR = inputs['GR']
        QS = inputs['QS']
        QI = inputs['QI']
        T = outputs['T']

        residuals['T'] = GL.dot(T)-GR*(T**4)+QS+QI
    def linearize(self, 
    inputs, outputs, partials):
        n = self.options['n'] + 1
        GL = inputs['GL']
        GR = inputs['GR']
        QS = inputs['QS']
        QI = inputs['QI']
        T = outputs['T']

        partials['T', 'GL'] = np.einsum('ij, k', np.eye(n, n), T)
        partials['T', 'GR'] = -np.diag(T**4)
        partials['T', 'QS'] = np.eye(n, n)
        partials['T', 'QI'] = np.eye(n, n)
        partials['T', 'T'] = GL - (4 * np.diag((GR * (T ** 3))))
    
    def guess_nonlinear(self, inputs, outputs, residuals):
        n = self.options['n'] + 1
        #gues values
        outputs['T'] = -np.ones(n)*50 + 273

if __name__ == "__main__":
    from inits import inits
    p = om.Problem()
    model = p.model

    nn = 13 #number of nodes in thermal model
    n = nn + 1

    nodes = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'
    GL, GR, QI, QS = inits(nn, nodes, conductors)
    GR = GR[:,0]

    #remove header row and column, as esatan base starts from 1
    #GL = GL[1:,1:]

    #define diagonal elements as negative of all node conductor couplings (sinks)
    diag = np.negative(np.sum(GL, 1))

    di = np.diag_indices(n)
    GL[di] = diag

    GL[0,0] = 1.0 # deep space node temperature = 0 K

    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('GL', val=np.zeros((n,n)), units='W/K')
    indeps.add_output('GR', val=np.zeros(n))
    indeps.add_output('QS', val=np.zeros(n), units='W')
    indeps.add_output('QI', val=np.zeros(n), units='W')
    model.add_subsystem('tmm', TempComp(n=nn), promotes=['*'])

    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 20
    model.linear_solver = om.DirectSolver()

    p.setup(check=True)


    p['GL'] = GL
    p['GR'] = GR
    p['QS'] = QS
    p['QI'] = QI

    
    p.run_model()
    print(p['T']-273.15)