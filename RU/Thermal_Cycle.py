import openmdao.api as om
import numpy as np

class SolarCell(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nodes', types=list, desc='list of input external surface node numbers')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        self.options.declare('alp_sc', default=.91, lower=.0, upper=1., desc='absorbtivity of the solar cell' )  
    def setup(self):
        nodes = self.options['nodes']
        n = len(nodes)
        m = self.options['npts']

        idx_list = [[(i,j) for j in range(m)] for i in nodes]

        self.add_input('T', val=np.ones((n,m))*28., src_indices=idx_list, units='degC')
        self.add_output('eta', val=np.ones((n,m))*0.3/0.91, desc='solar cell efficiency with respect to absorbed power for input surface nodes over time ')
        self.declare_partials('*', '*')
    def compute(self, inputs, outputs):
        """solar cell data from:https://www.e3s-conferences.org/articles/e3sconf/pdf/2017/04/e3sconf_espc2017_03011.pdf"""
        
        alp_sc = self.options['alp_sc']
        
        T0 = 28. #reference temperature
        eff0 = .285 #efficiency at ref temp
        T1 = -150.
        eff1 = 0.335

        delta_T = inputs['T'] - T0

        slope = (eff1 - eff0) / (T1 - T0)
        
        outputs['eta'] = (eff0 + slope * delta_T)/alp_sc

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        
        alp_sc = self.options['alp_sc']
        T0 = 28.
        eff0 = .285
        T1 = -150.
        eff1 = 0.335
        slope = (eff1 - eff0) / (T1 - T0)
        
        deff_dT = d_outputs['eta']

        if mode == 'fwd':
            
            deff_dT += slope * d_inputs['T'] / alp_sc
        else:

            d_inputs['T'] = slope * deff_dT / alp_sc

class ElectricPower(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nodes', types=list, desc='list of input external surface node numbers')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        self.options.declare('ar', default=.9, lower=.0, upper=1., desc='solar cell to node surface area ratio')
        self.options.declare('eta_con', default=.95, lower=.0, upper=1., desc='MPPT converter efficiency')

    def setup(self):
        n = len(self.options['nodes'])
        m = self.options['npts']
    
        self.add_input('eta', val=np.ones((n,m))*0.3/0.91, desc='solar cell efficiency with respect to absorbed power for input surface nodes over time ')
        self.add_input('QS_c', shape=(n,m), desc='solar cell absorbed power over time', units='W')
        self.add_output('P_el', shape=(n,m), desc='Electrical power output over time', units='W')
        #self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        ar = self.options['ar']
        m = self.options['npts']
        n = len(self.options['nodes'])
        eta_con = self.options['eta_con']
    
        eta = inputs['eta'] * eta_con * ar
        QS = inputs['QS_c']

        outputs['P_el'] = np.multiply(QS, eta)

        """ def compute_partials(self, input, partials):
        rows = self.options['n_in']
        cols = self.options['npts']
        partials['P_el', 'QS_c'] = np.einsum('ik, jl', np.eye(cols, cols), np.eye(rows, rows))
        partials['P_el', 'eta'] = np.einsum('ik, jl', np.eye(cols, cols), np.eye(rows, rows)) """

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        eta_con = self.options['eta_con']
        ar = self.options['ar']
        eta = inputs['eta'] * eta_con * ar

        dP_el = d_outputs['P_el']

        if mode == 'fwd':
            if 'QS_c' in d_inputs:
                
                dP_el += d_inputs['QS_c'] * eta

            if 'eta' in d_inputs:
                
                dP_el += d_inputs['eta'] * inputs['QS_c'] * eta_con * ar
        else:
            
            if 'QS_c' in d_inputs:
                d_inputs['QS_c'] += dP_el * eta

            if 'eta' in d_inputs:
                d_inputs['eta'] += inputs['QS_c'] * eta_con * ar * dP_el

class QSmtxComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nn', types=int, desc='number of diffusion nodes in thermal model')
        self.options.declare('nodes', types=list, desc='list of input external surface node numbers')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        
    def setup(self):
        nn = self.options['nn'] + 1
        n = len(self.options['nodes'])
        m = self.options['npts']
        self.add_input('P_el', shape=(n,m), desc='solar cell electric power over time', units='W')
        self.add_input('QS_c', shape=(n,m), desc='solar cell absorbed heat over time', units='W')
        self.add_input('QS_r', shape=(n,m), desc='radiator absorbed heat over time', units='W')
        self.add_output('QS', val=np.zeros((nn,m)), desc='solar absorbed heat over time', units='W')
    
    def compute(self, inputs, outputs):
        nn = self.options['nn'] + 1
        m = self.options['npts']
        QS = np.zeros((nn,m))
        P_el = inputs['P_el']
        QS_c = inputs['QS_c']
        QS_r = inputs['QS_r']
        for i,node in enumerate(self.options['nodes']):
            QS[node,:] = QS_c[i,:] + QS_r[i,:] - P_el[i,:] # energy balance
        outputs['QS'] = QS
    
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        m = self.options['npts']
        nodes = self.options['nodes']
                
        P_el = inputs['P_el']
        QS_c = inputs['QS_c']
        QS_r = inputs['QS_r']

        dQS = d_outputs['QS']

        if mode == 'fwd':
            
            if 'P_el' in d_inputs:
                for i,node in enumerate(nodes):
                    dQS[node,:] -= d_inputs['P_el'][i,:]

            if 'QS_c' in d_inputs:
                for i,node in enumerate(nodes):
                    dQS[node,:] += d_inputs['QS_c'][i,:]
            
            if 'QS_r' in d_inputs:
                for i,node in enumerate(nodes):
                    dQS[node,:] += d_inputs['QS_r'][i,:]
        else:

            if 'P_el' in d_inputs:
                for i,node in enumerate(nodes):
                    d_inputs['P_el'][i,:] -= dQS[node,:]

            if 'QS_c' in d_inputs:
                for i,node in enumerate(nodes):
                    d_inputs['QS_c'][i,:] += dQS[node,:]
            
            if 'QS_r' in d_inputs:
                for i,node in enumerate(nodes):
                    d_inputs['QS_r'][i,:] += dQS[node,:]

class TempsComp(om.ImplicitComponent):
    """Computes steady state node temperatures over multiple points."""
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
        self.declare_partials(of='T', wrt='T', method='fd')
        self.declare_partials(of='T', wrt='GL')
        self.declare_partials(of='T', wrt='GR')
        self.declare_partials(of='T', wrt='QI')
        self.declare_partials(of='T', wrt='QS')

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
        #partials['T', 'T'] = np.einsum('ik, jl', GL, np.eye(m, m)) + np.einsum('ik, jl', GR, np.eye(m, m))
    
    def guess_nonlinear(self, inputs, outputs, residuals):
        n = self.options['n'] + 1
        m = self.options['npts']
        #gues values
        outputs['T'] = -np.ones((n,m))*50 + 273

class Thermal_Cycle(om.Group):
    def __init__(self, nn, npts, nodes):
            super(Thermal_Cycle, self).__init__()

            self.nn = nn
            self.npts = npts
            self.nodes = nodes

    def setup(self):
    
        nodes = self.nodes

        self.add_subsystem('sc', SolarCell(nodes=nodes, npts=self.npts), promotes=['*'])
        self.add_subsystem('el', ElectricPower(nodes=nodes, npts=self.npts), promotes=['*'])
        self.add_subsystem('QS', QSmtxComp(nn=self.nn, nodes=nodes, npts=self.npts), promotes=['*'])
        self.add_subsystem('temps', TempsComp(n=self.nn, npts=self.npts), promotes=['*'])
        
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 15
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.linesearch.options['maxiter'] = 10
        self.nonlinear_solver.linesearch.options['iprint'] = 2
        self.linear_solver = om.DirectSolver()
        self.linear_solver.options['assemble_jac'] = False

if __name__ == "__main__":

    from Pre_process import parse_vf, inits, conductors, nodes, idx_dict, opticals
    from Solar import Solar
    
    npts = 2

    nn, groups = nodes()
    GL_init, GR_init = conductors(nn=nn)
    QI_init1, QS_init1 = inits()
    nodes2 = 'Nodal_data_2.csv'
    QI_init2, QS_init2 = inits(nodes2)

    npts = 2

    QI_init = np.concatenate((QI_init1, QI_init2), axis=1)
    #QS_init = np.concatenate((QS_init1, QS_init2), axis=1)

    optprop = parse_vf()

    #keys = list(groups.keys()) # import all nodes?
    keys = ['Box:outer', 'Panel_outer:solar_cells', 'Panel_inner:solar_cells', 'Panel_body:solar_cells']
    faces = opticals(groups, keys, optprop)    

    #compute total number of nodes in selected faces
    nodes = []
    for face in faces:
        nodes.extend(face['nodes'])

    """ data = parse_vf()
    nodes = list(data.keys()) """

    # index dictionary
    idx = idx_dict(nodes, groups)

    # indices for solar cells
    sc_idx = sum([idx[keys] for keys in ['Panel_outer:solar_cells', 'Panel_inner:solar_cells', 'Panel_body:solar_cells']], [])

    model = Thermal_Cycle(nn=nn, npts=npts, nodes=nodes)
    
    params = model.add_subsystem('params', om.IndepVarComp(), promotes=['*'])
    model.add_subsystem('sol', Solar(npts=npts, n_in = len(nodes), faces=faces), promotes=['*'])
    params.add_output('QI', val=QI_init)
    params.add_output('GL', val=GL_init, units='W/K')
    params.add_output('GR', val=GR_init)
    params.add_output('phi', val=np.array([10.,10.]) )
    params.add_output('dist', val=np.array([3., 1.]))
    params.add_output('cr', val=np.ones((len(nodes), 1)))
    params.add_output('alp_r', val=np.ones((len(nodes), 1)))
    
    problem = om.Problem(model=model)
    problem.setup(check=True)
    
    problem['cr'][sc_idx] = 1.0
    problem['alp_r'][list(idx['Box:outer'])] = 0.5

    problem.run_model()

    print(problem['T']-273.)
    #print(problem['eta'])
    #print(problem['QS'][1:12,:] - QS_init[1:12,:])
    #print(GR_init1 == GR_init2)

    check_partials_data = problem.check_partials(compact_print=True, show_only_incorrect=False, form='forward', step=1e-03)

    """ totals = problem.compute_totals(of=['T'], wrt=['eta'])
    print(totals)
    problem.check_totals(compact_print=True) """
