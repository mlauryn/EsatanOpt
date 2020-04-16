import openmdao.api as om
import numpy as np
from TempsComp import TempsComp

class ElectricPower(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nodes', desc='list of input external surface node numbers')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        self.options.declare('ar', default=1., lower=.0, upper=1., desc='solar cell to node surface area ratio')
        self.options.declare('eta_con', default=1., lower=.0, upper=1., desc='MPPT converter efficiency')

    def eta_in(self): # input path efficiency (mppt losses * area losses)
        ar = self.options['ar']
        eta_con = self.options['eta_con']
        eta_in = eta_con * ar
        return eta_in

    def setup(self):
        n = len(self.options['nodes'])
        m = self.options['npts']
    
        self.add_input('eta', val=np.ones((n,m))*0.3/0.91, desc='solar cell efficiency with respect to absorbed power for input surface nodes over time ')
        self.add_input('QS_c', shape=(n,m), desc='solar cell absorbed power over time', units='W')
        self.add_output('P_el', shape=(n,m), desc='Electrical power output over time', units='W')
        rows = np.arange(n*m)
        cols = rows
        self.declare_partials('P_el', 'eta', rows=rows, cols=cols)
        self.declare_partials('P_el', 'QS_c', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        
        eta = inputs['eta'] * self.eta_in() # total efficiency = cell eff * input path eff
        QS = inputs['QS_c']

        outputs['P_el'] = np.multiply(QS, eta)

    def compute_partials(self, inputs, partials):
        
        n = len(self.options['nodes'])
        m = self.options['npts']
        partials['P_el','eta'] = (inputs['QS_c'] * self.eta_in()).reshape(n*m,)
        partials['P_el', 'QS_c'] = (inputs['eta'] * self.eta_in()).reshape(n*m,)

class QSmtxComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nn', types=int, desc='number of diffusion nodes in thermal model')
        self.options.declare('nodes', desc='list of input external surface node numbers')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        
    def setup(self):
        nn = self.options['nn'] + 1
        nodes = self.options['nodes']
        n = len(nodes)
        m = self.options['npts']
        self.add_input('P_el', shape=(n,m), desc='solar cell electric power over time', units='W')
        self.add_input('QS_c', shape=(n,m), desc='solar cell absorbed heat over time', units='W')
        self.add_input('QS_r', shape=(n,m), desc='radiator absorbed heat over time', units='W')
        self.add_output('QS', val=np.zeros((nn,m)), desc='solar absorbed heat over time', units='W')
        
        y = np.arange(nn*m).reshape((nn,m))
        rows = y[nodes,:].flatten()
        cols = np.arange(n*m)
        self.declare_partials('QS', 'P_el', rows=rows, cols=cols, val=-1.)
        self.declare_partials('QS', 'QS_c', rows=rows, cols=cols, val=1.)
        self.declare_partials('QS', 'QS_r', rows=rows, cols=cols, val=1.)
    
    def compute(self, inputs, outputs):
        nn = self.options['nn'] + 1
        m = self.options['npts']
        QS = np.zeros((nn,m))
        P_el = inputs['P_el']
        QS_c = inputs['QS_c']
        QS_r = inputs['QS_r']
        nodes = self.options['nodes']
        for i,node in enumerate(nodes):
            QS[node,:] = QS_c[i,:] + QS_r[i,:] - P_el[i,:] # energy balance
        outputs['QS'] = QS

    def compute_partials(self, inputs, partials):
        pass

class Thermal_direct(om.Group):
    def __init__(self, nn, npts, nodes):
            super(Thermal_direct, self).__init__()

            self.nn = nn
            self.npts = npts
            self.nodes = nodes

    def setup(self):
    
        nodes = self.nodes

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

    from Pre_process import parse_vf, parse_cond, inits, conductors, nodes, idx_dict, opticals
    from Solar import Solar
    
    npts = 2

    nn, groups = nodes(data='nodes_RU_v4_base_cc.csv')
    GL_init, GR_init = conductors(nn=nn, data='cond_RU_v4_base_cc.csv')

    cond_data = parse_cond(filepath='links_RU_v4_base.txt') 
    optprop = parse_vf(filepath='vf_RU_v4_base.txt')

    QI_init1, QS_init1 = inits(data='nodes_RU_v4_base_cc.csv')
    QI_init2, QS_init2 = inits(data='nodes_RU_v4_base_hc.csv')

    QI_init = np.concatenate((QI_init1, QI_init2), axis=1)
    #QS_init = np.concatenate((QS_init1, QS_init2), axis=1)

    #keys = list(groups.keys()) # import all nodes?
    keys = ['Box', 'Panel_outer', 'Panel_inner', 'Panel_body']
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

    model = Thermal_direct(nn=nn, npts=npts, nodes=nodes)
    
    params = model.add_subsystem('params', om.IndepVarComp(), promotes=['*'])
    model.add_subsystem('sol', Solar(npts=npts, n_in = len(nodes), faces=faces, model='RU_v4_base'), promotes=['*'])
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

    #print(problem['T']-273.)
    #print(problem['eta'])
    #print(problem['QS'][1:12,:] - QS_init[1:12,:])
    #print(GR_init1 == GR_init2)

    check_partials_data = problem.check_partials(compact_print=True, show_only_incorrect=False, form='forward', step=1e-03)

    """ totals = problem.compute_totals(of=['T'], wrt=['eta'])
    print(totals)
    problem.check_totals(compact_print=True) """
    
