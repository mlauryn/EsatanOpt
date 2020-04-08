import openmdao.api as om
import numpy as np
import math

class Ilumination(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('G_sc', default=1365.0, desc='solar constant')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        self.options.declare('A', default=[1]*7, types=list, desc='input surface node areas')
    def setup(self):
        n = len(self.options['A'])
        m = self.options['npts']
        self.add_input('dist', val=np.ones(m), desc='distane in AU')
        self.declare_partials(of='QIS', wrt='dist', dependent=False) # distance will be independat variable
        self.add_input('beta', val=np.ones(m), units='rad')
        self.add_output('QIS', val=np.ones((n,m)), units='W')

    def compute(self, inputs, outputs):
        m = self.options['npts']
        n = len(self.options['A'])
        d = inputs['dist']
        Gsc = self.options['G_sc']
        
        A = self.options['A']
        beta = inputs['beta']
        QIS = np.zeros((n,m))

        q_s = Gsc * (1/d)**2 #solar flux at distance d

        for i in range(m):
            for n in [0,2,3]: #these nodes are solar cells
                QIS[n,i] = q_s[i] * A[n] * np.cos(beta[i])
            QIS[5,i] = q_s[i] * A[5] * np.sin(beta[i])
        outputs['QIS'] = QIS

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        m = self.options['npts']
        d = inputs['dist']
        beta = inputs['beta']
        A = self.options['A']
        Gsc = self.options['G_sc']
        q_s = Gsc * (1/d)**2 #solar flux at distance d

        dQIS = d_outputs['QIS']

        if mode == 'fwd':
            
            if 'beta' in d_inputs:
                for i in range(m):
                    for n in [0,2,3]: #these nodes are solar cells
                        dQIS[n,i] -= q_s[i] * A[n] *  np.sin(beta[i]) * d_inputs['beta'][i]
                    dQIS[5,i] += q_s[i] * A[5] * np.cos(beta[i]) * d_inputs['beta'][i]
        else:
            
            if 'beta' in d_inputs:
                for i in range(m):
                    for n in [0,2,3]: #these nodes are solar cells
                        d_inputs['beta'][i] -= q_s[i] * A[n] * np.sin(beta[i]) * dQIS[n,i]
                    d_inputs['beta'][i] += q_s[i] * A[5] * np.cos(beta[i]) * dQIS[5,i]

class SolarPower(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n_in', types=int, desc='number of input nodes')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        #self.options.declare('eta_con', default=.95, lower=.0, upper=1., desc='MPPT converter efficiency')
        self.options.declare('alp_sc', default=.91, lower=.0, upper=1., desc='absorbtivity of the solar cell' )

    def setup(self):
        n = self.options['n_in']
        m = self.options['npts']
        
        self.add_input('QIS', shape=(n,m), desc='incident solar power', units='W')
        self.add_input('alp_r', shape=(n,1), desc='absorbtivity of the input node radiating surface')
        self.add_input('cr', shape=(n,1), desc='solar cell or radiator installation decision for input nodes')
        self.add_output('QS_c', shape=(n,m), desc='solar cell absorbed power over time', units='W')
        self.add_output('QS_r', shape=(n,m), desc='radiator absorbed power over time', units='W')

    def compute(self, inputs, outputs):
       
        alp_sc = self.options['alp_sc']

        QIS = inputs['QIS']
        alp_r = inputs['alp_r']
        cr = inputs['cr']
        
        QS_c = QIS * alp_sc * cr
        QS_r = QIS * alp_r * (1 - cr)

        outputs['QS_c'] = QS_c
        outputs['QS_r'] = QS_r

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
    def __init__(self, npts, area):
            super(Solar, self).__init__()

            self.npts = npts # number of points
            self.area = area #area of optical nodes

    def setup(self):
        
        npts = self.npts
        area = self.area
        n_in = len(area)

        self.add_subsystem('il', Ilumination(npts=npts, A=area), promotes=['*'])
        self.add_subsystem('sol', SolarPower(n_in=n_in, npts=npts), promotes=['*'])

if __name__ == "__main__":

    from ViewFactors import parse_vf
    from inits import inits
    
    npts = 2

    view_factors = 'viewfactors.txt'
    data = parse_vf(view_factors)

    area = []
    for entry in data:
        area.append(entry['area'])
    #print(nodes, area, vf, eps)
    
    params = om.IndepVarComp()
    params.add_output('beta', val=np.zeros(npts) )
    params.add_output('dist', val=[1., 3.])

    model = Solar(npts=npts, area=area)

    model.add_subsystem('params', params, promotes=['*'])
    
    problem = om.Problem(model=model)
    problem.setup(check=True)
    
    problem.run_model()
    
    #check_partials_data = problem.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-02)

    #compare results with esatan
    nodes = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'
    n, GL_init1, GR_init1, QI_init1, QS_init1 = inits(nodes, conductors)
    nodes2 = 'Nodal_data_2.csv'
    conductors2 = 'Cond_data_2.csv'
    n, GL_init2, GR_init2, QI_init2, QS_init2 = inits(nodes2, conductors2)
    npts = 2

    QS_init = np.concatenate((QS_init2, QS_init1), axis=1)
    

    print((problem['QS_c'] - QS_init[1:12,:]*0.91/0.61)/problem['QS_c'])
    #print(QS_init[1:12,:]*0.91/0.61)
    
    
    #problem.model.list_inputs(print_arrays=True)