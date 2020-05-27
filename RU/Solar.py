import openmdao.api as om
import numpy as np
from HeatFluxComp import HeatFluxComp

class Incident_Solar(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('npts', default=1, types=int, desc='number of points')
        self.options.declare('areas', desc='1D array of areas of surface nodes')
    def setup(self):
        n = len(self.options['areas'])
        m = self.options['npts']
        self.add_input('dist', val=np.ones(m), desc='distane in AU')
        self.add_input('q_s', val=np.zeros((m,n)))
        self.add_output('QIS', val=np.ones((n, m)), units='W')

        self.A = self.options['areas']

        rows = np.arange(n*m)
        cols1 = np.arange(m)[np.newaxis].repeat(n,axis=0).flatten()
        cols2 = np.arange(n*m).reshape((m,n)).flatten(order='F')
        self.declare_partials(of='QIS', wrt='dist', rows=rows, cols=cols1)
        self.declare_partials(of='QIS', wrt='q_s', rows=rows, cols=cols2)

    def compute(self, inputs, outputs):

        n = len(self.options['areas'])
        m = self.options['npts']
        d = inputs['dist']
        q_s = inputs['q_s']
        
        QIS = np.zeros((n,m))
        for i in range(m):
            QIS[:,i] = q_s[i,:] * self.A * d[i]**(-2) # incident solar power at distance d at each point

        outputs['QIS'] = QIS 

    def compute_partials(self, inputs, partials):

        n = len(self.options['areas'])
        partials['QIS', 'dist'] = (-2 * inputs['q_s'] * (inputs['dist'][:,np.newaxis])**(-3) * self.A).flatten('F')
        partials['QIS', 'q_s'] = (((inputs['dist'][:,np.newaxis]).repeat(n,axis=1))**(-2) * self.A).flatten('F')

class SolarPower(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n_in', types=int, desc='number of input nodes')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        self.options.declare('alp_sc', default=.91, lower=.0, upper=1., desc='absorbtivity of the solar cell' )

    def setup(self):
        n = self.options['n_in']
        m = self.options['npts']
        
        self.add_input('QIS', shape=(n,m), desc='incident solar power', units='W')
        self.add_input('alp_r', shape=(n,1), desc='solar absorbtivity of the input node radiating surface')
        self.add_input('cr', shape=(n,1), desc='solar cell or radiator installation decision for input nodes')
        self.add_output('QS_c', shape=(n,m), desc='solar cell absorbed power over time', units='W')
        self.add_output('QS_r', shape=(n,m), desc='radiator absorbed power over time', units='W')

        rows = np.arange(n*m)
        cols1 = rows
        cols2 = np.arange(n).repeat(m)

        self.declare_partials(of='QS*', wrt='QIS', rows=rows, cols=cols1)
        self.declare_partials(of='QS*', wrt='cr', rows=rows, cols=cols2)
        self.declare_partials(of='QS_r', wrt='alp_r', rows=rows, cols=cols2)
        self.declare_partials(of='QS_c', wrt='alp_r', dependent=False)

    def compute(self, inputs, outputs):
       
        alp_sc = self.options['alp_sc']

        QIS = inputs['QIS']
        alp_r = inputs['alp_r']
        cr = inputs['cr']
        
        outputs['QS_c'] = QIS * alp_sc * cr
        outputs['QS_r'] = QIS * alp_r * (1 - cr)

    def compute_partials(self, inputs, partials):

        alp_sc = self.options['alp_sc']
        m = self.options['npts']
        
        partials['QS_c', 'QIS'] = alp_sc * inputs['cr'].repeat(m)
        partials['QS_r', 'QIS'] = (inputs['alp_r'] * (1-inputs['cr'])).repeat(m)
        partials['QS_c', 'cr'] = alp_sc * inputs['QIS'].flatten()
        partials['QS_r', 'cr'] = (-inputs['alp_r'] * inputs['QIS']).flatten()
        partials['QS_r', 'alp_r'] = ((1 - inputs['cr']) * inputs['QIS']).flatten()

class Solar(om.Group):
    def __init__(self, npts, areas, nodes, model):
            super(Solar, self).__init__()

            self.npts = npts # number of points
            self.areas = areas # areas input nodes
            self.nodes = nodes # input node numbers
            self.n = len(nodes) # number of input external surface nodes
            self.model = model #Esatan radiative model name


    def setup(self):

        self.add_subsystem('hf', HeatFluxComp(nodes=self.nodes, npts=self.npts, model=self.model), promotes=['*'])
        self.add_subsystem('is', Incident_Solar(npts=self.npts, areas=self.areas), promotes=['*'])
        self.add_subsystem('sol', SolarPower(n_in=self.n, npts=self.npts), promotes=['*'])

if __name__ == "__main__":

    from Pre_process import parse_vf, opticals, nodes, inits, idx_dict

    model_name = 'RU_v4_base'
    nn, groups = nodes(data='./Esatan_models/'+model_name+'/nodes_output.csv')

    optprop = parse_vf(filepath='./Esatan_models/'+model_name+'/vf_report.txt')

    #keys = list(groups.keys()) # import all nodes?
    keys = ['Box', 'Panel_body']
    faces = opticals(groups, keys, optprop)    

    #compute total number of nodes in selected faces
    nodes = []
    areas = []
    for face in faces:
        nodes.extend(face['nodes'])
        areas.extend(face['areas'])
    n_in = len(nodes)

    idx = idx_dict(nodes, groups)
    
    npts = 2

    params = om.IndepVarComp()
    params.add_output('phi', val=np.array([10.,10.]) )
    params.add_output('dist', val=np.array([3., 1.]))
    params.add_output('cr', val=np.ones((n_in, 1)))
    params.add_output('alp_r', val=np.ones((n_in, 1)))

    model = Solar(npts=npts, nodes=nodes, areas=areas, model=model_name)

    model.add_subsystem('params', params, promotes=['*'])
    
    problem = om.Problem(model=model)
    problem.setup(check=True)

    #assign initial values

    problem['cr'][list(idx['Box'])] = 0.0
    problem['alp_r'][list(idx['Box'])] = 0.5    

    problem.run_model()
    
    check_partials_data = problem.check_partials(compact_print=True, show_only_incorrect=False, method='cs')

    #compare results with esatan
    """ QI_init1, QS_init1 = inits(data='nodes_RU_v4_base_cc.csv')
    QI_init2, QS_init2 = inits(data='nodes_RU_v4_base_hc.csv')
    
    npts = 2

    QS_init = np.concatenate((QS_init1, QS_init2), axis=1) """
    
    #check relative error
    """ print((problem['QS_c'] - QS_init[nodes,:]*0.91/0.61)/problem['QS_c'])  
    print(QS_init[nodes,:]*0.91/0.61) 
    print(problem['QS_c']) """

    """ print((problem['QS_r'] - QS_init[nodes,:])/problem['QS_r'])
    print(QS_init[nodes,:])
    print(problem['QS_r'])

    print(nodes) """

    #problem.model.list_inputs(print_arrays=True)

