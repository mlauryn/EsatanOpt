import openmdao.api as om
import numpy as np
from HeatFluxComp import HeatFluxComp

class Incident_Solar(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_in', types=int, desc='number of input nodes')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        self.options.declare('faces', types=list, desc='names and optical properties of input faces')
    def setup(self):
        faces = self.options['faces']
        n = self.options['n_in']
        m = self.options['npts']
        self.add_input('dist', val=np.ones(m), desc='distane in AU')
        self.add_input('q_s', val=np.zeros((m,n)))
        self.add_output('QIS', val=np.ones((n, m)), units='W')

        area = [] # compute area of each node
        for face in faces:
            area.extend(face['areas'])
        self.A = np.array(area)

    def compute(self, inputs, outputs):

        n = self.options['n_in']
        m = self.options['npts']
        d = inputs['dist']
        q_s = inputs['q_s']
        
        QIS = np.zeros((n,m))
        for i in range(m):
            QIS[:,i] = q_s[i,:] * self.A * d[i]**(-2) # incident solar power at distance d at each point

        outputs['QIS'] = QIS 

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        m = self.options['npts']
        n = self.options['n_in']
        d = inputs['dist']
        q_s = inputs['q_s']

        dQIS = d_outputs['QIS']

        if mode == 'fwd':
            
            if 'q_s' in d_inputs:

                for i in range(m):
                    dQIS[:,i] += self.A * d[i]**(-2) * d_inputs['q_s'][i,:]

            if 'dist' in d_inputs:
                for i in range(m):
                    dQIS[:,i] -= q_s[i,:] * self.A * 2 * d[i]**(-3) * d_inputs['dist'][i]

        else:
            
            if 'q_s' in d_inputs:

                for i in range(m):
                    d_inputs['q_s'][i,:] += self.A * d[i]**(-2) * dQIS[:,i]

            if 'dist' in d_inputs:

                for i in range(m):
                    for k in range(n):
                        d_inputs['dist'][i] -= q_s[i,k] * self.A[k] * 2 * d[i]**(-3) * dQIS[k,i]

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

    def compute(self, inputs, outputs):
       
        alp_sc = self.options['alp_sc']

        QIS = inputs['QIS']
        alp_r = inputs['alp_r']
        cr = inputs['cr']
        
        outputs['QS_c'] = QIS * alp_sc * cr
        outputs['QS_r'] = QIS * alp_r * (1 - cr)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        m = self.options['npts']
        alp_sc = self.options['alp_sc']
        QIS = inputs['QIS']
        alp_r = inputs['alp_r']
        cr = inputs['cr']

        dQSc = d_outputs['QS_c']
        dQSr = d_outputs['QS_r']        

        if mode == 'fwd':
            if 'QIS' in d_inputs:
                
                dQSc += d_inputs['QIS'] * alp_sc * cr
                dQSr += d_inputs['QIS'] * alp_r * (1 -cr)

            if 'alp_r' in d_inputs:
                
                dQSc += 0.0
                dQSr += QIS * d_inputs['alp_r']* (1 - cr)

            if 'cr' in d_inputs:
                
                dQSc += QIS * alp_sc * d_inputs['cr']
                dQSr -= QIS * alp_r * d_inputs['cr']
        else:
            
            if 'QIS' in d_inputs:
                
                d_inputs['QIS'] += dQSc * alp_sc * cr
                d_inputs['QIS'] += dQSr * alp_r * (1 -cr)

            if 'alp_r' in d_inputs:
                for i in range(m):
                    d_inputs['alp_r'] += (QIS[:,i] * dQSr[:,i])[np.newaxis].T * (1 - cr)

            if 'cr' in d_inputs:
                for i in range(m):
                    d_inputs['cr'] += (QIS[:,i] * dQSc[:,i])[np.newaxis].T * alp_sc
                    d_inputs['cr'] -= alp_r * (dQSr[:,i] * QIS[:,i])[np.newaxis].T

class Solar(om.Group):
    def __init__(self, npts, n_in, faces):
            super(Solar, self).__init__()

            self.npts = npts # number of points
            self.faces = faces # optical properties of input faces
            self.n = n_in # number of input external surface nodes 

    def setup(self):

        self.add_subsystem('hf', HeatFluxComp(faces=self.faces, npts=self.npts), promotes=['*'])
        self.add_subsystem('is', Incident_Solar(npts=self.npts, n_in=self.n, faces=self.faces), promotes=['*'])
        self.add_subsystem('sol', SolarPower(n_in=self.n, npts=self.npts), promotes=['*'])

if __name__ == "__main__":

    from Pre_process import parse_vf, opticals, nodes, inits, idx_dict

    nn, groups = nodes()

    optprop = parse_vf()

    #keys = list(groups.keys()) # import all nodes?
    keys = ['Box:outer', 'Panel_inner:solar_cells', ]
    faces = opticals(groups, keys, optprop)    

    #compute total number of nodes in selected faces
    nodes = []
    for face in faces:
        nodes.extend(face['nodes'])
    n_in = len(nodes)

    idx = idx_dict(nodes, groups)
    
    npts = 2

    params = om.IndepVarComp()
    params.add_output('phi', val=np.array([10.,10.]) )
    params.add_output('dist', val=np.array([3., 1.]))
    params.add_output('cr', val=np.ones((n_in, 1)))
    params.add_output('alp_r', val=np.ones((n_in, 1)))

    model = Solar(npts=npts, n_in = n_in, faces=faces)

    model.add_subsystem('params', params, promotes=['*'])
    
    problem = om.Problem(model=model)
    problem.setup(check=True)

    #assign initial values

    problem['cr'][list(idx['Box:outer'])] = 0.0
    problem['alp_r'][list(idx['Box:outer'])] = 0.5    

    problem.run_model()
    
    #check_partials_data = problem.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-02)

    #compare results with esatan
    QI_init1, QS_init1 = inits()
    QI_init2, QS_init2 = inits(data='Nodal_data_2.csv')
    
    npts = 2

    QS_init = np.concatenate((QS_init1, QS_init2), axis=1)
    
    #check relative error
    """ print((problem['QS_c'] - QS_init[nodes,:]*0.91/0.61)/problem['QS_c'])  
    print(QS_init[nodes,:]*0.91/0.61) 
    print(problem['QS_c']) """

    print((problem['QS_r'] - QS_init[nodes,:])/problem['QS_r'])
    print(QS_init[nodes,:])
    print(problem['QS_r'])

    print(nodes)

    #problem.model.list_inputs(print_arrays=True)

