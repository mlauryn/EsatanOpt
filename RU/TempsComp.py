import openmdao.api as om
import numpy as np

class TempsComp(om.ImplicitComponent):
    """Computes steady state node temperatures over multiple design points."""
    def initialize(self):
        self.options.declare('n', default=1, types=int, desc='number of diffusion nodes')
        self.options.declare('npts', default=1, types=int, desc='number of points')
    def setup(self):
        n = self.options['n'] + 1
        m = self.options['npts']
        self.add_output('T', val=np.zeros((n,m)), units='K')
        self.add_input('GL', val=np.zeros((n,n)), units='W/K')
        self.add_input('GR', val=np.zeros((n,n)))
        self.add_input('QS', val=np.zeros((n,m)), units='W')
        self.add_input('QI', val=np.zeros((n,m)), units='W')
        self.declare_partials(of='T', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        GL = inputs['GL']
        GR = inputs['GR']
        QS = inputs['QS']
        QI = inputs['QI']
        T = outputs['T']

        residuals['T'] = GL.dot(T) + GR.dot(T**4) + QS + QI

    def linearize(self, inputs, outputs, partials):
        n = self.options['n'] + 1
        m = self.options['npts']
        GL = inputs['GL']
        GR = inputs['GR']
        QS = inputs['QS']
        QI = inputs['QI']
        T = outputs['T']

        partials['T', 'GL'] = np.einsum('ik, jl', np.eye(n, n), T.T)
        partials['T', 'GR'] = np.einsum('ik, jl', np.eye(n, n), (T**4).T)
        partials['T', 'QS'] = np.einsum('ik, jl', np.eye(n, n), np.eye(m, m))
        partials['T', 'QI'] = np.einsum('ik, jl', np.eye(n, n), np.eye(m, m))
        partials['T', 'T'] = np.einsum('ik, jl', GL, np.eye(m, m)) + 4.0 * T**3 * np.einsum('ik, jl', GR, np.eye(m, m))
    
    def guess_nonlinear(self, inputs, outputs, residuals):
        n = self.options['n'] + 1
        m = self.options['npts']
        #gues values
        outputs['T'] = -np.ones((n,m))*50 + 273

if __name__ == "__main__":
    from Pre_process import nodes, conductors, inits
    problem = om.Problem()
    model = problem.model

    nn, groups = nodes(data='nodes_RU_v4_detail_cc.csv')
    GL_init, GR_init = conductors(nn=nn, data='cond_RU_v4_detail_hc.csv')
    QI_init1, QS_init1 = inits(data='nodes_RU_v4_detail_cc.csv')
    QI_init2, QS_init2 = inits(data='nodes_RU_v4_detail_hc.csv')

    npts = 2

    QI_init = np.concatenate((QI_init1, QI_init2), axis=1)
    QS_init = np.concatenate((QS_init1, QS_init2), axis=1)

    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('GL', val=GL_init, units='W/K')
    indeps.add_output('GR', val=GR_init)
    indeps.add_output('QS', val=QS_init, units='W')
    indeps.add_output('QI', val=QI_init, units='W')

    model.add_subsystem('tmm', TempsComp(n=nn, npts=npts), promotes=['*'])

    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 50
    """ model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
    model.nonlinear_solver.linesearch.options['maxiter'] = 10
    model.nonlinear_solver.linesearch.options['iprint'] = 2 """
    model.linear_solver = om.DirectSolver()

    problem.setup(check=True)

    problem.run_model()
    print(problem['T']-273.15)

    #check_partials_data = problem.check_partials(compact_print=True, show_only_incorrect=True, form='central', step=1e-4)
    #problem.model.list_inputs(print_arrays=True, includes=['*QI*'])