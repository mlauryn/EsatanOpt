import openmdao.api as om
import numpy as np

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
                        dQIS[n,i] -= q_s[i] * A[n] *  np.sin(beta[i]) * d_inputs['beta']
                    dQIS[5,i] += q_s[i] * A[5] * np.cos(beta[i]) * d_inputs['beta']
        else:
            
            if 'beta' in d_inputs:
                for i in range(m):
                    for n in [0,2,3]: #these nodes are solar cells
                        d_inputs['beta'] -= q_s[i] * A[n] * np.sin(beta[i]) * dQIS[n,i]
                    d_inputs['beta'] += q_s[i] * A[5] * np.cos(beta[i]) * dQIS[5,i]

if __name__ == "__main__":
    #debug script:
        
    
    prob = om.Problem()
    model = prob.model

    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('beta', val=30., units='deg')
    
    model.add_subsystem('QIS', Ilumination(), promotes=['*'])

    prob.setup(check=True)

    prob.run_model()

    check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=True, form='central', step=1e-05)

    print(prob['QIS'])