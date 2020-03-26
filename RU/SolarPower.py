import openmdao.api as om
import numpy as np

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

if __name__ == "__main__":
    #debug script:
        
    
    npts = 3
    n = 5
    
    alp_sc = 1.0

    prob = om.Problem()
    model = prob.model

    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    #indeps.add_output('eta', val=np.ones((n_in, npts))*0.3)
    indeps.add_output('QIS', val=np.ones((n,npts)))
    indeps.add_output('alp_r', val=np.ones((n,1))*0.1)
    indeps.add_output('cr', val=np.ones((n,1))*.5)
    
    model.add_subsystem('p', SolarPower(n_in=n, npts=npts, alp_sc=alp_sc), promotes=['*'])

    prob.setup(check=True)

    prob.run_model()

    check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=True, form='central', step=1e-02)

    print(prob['QS_c'])
    print(prob['QS_r'])