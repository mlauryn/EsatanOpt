import openmdao.api as om
import numpy as np

class PowerInput(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_in', types=int, desc='number of input nodes')
        self.options.declare('npts', default=1, types=int, desc='number of points')
    def setup(self):
        n = self.options['n_in']
        m = self.options['npts']
        self.add_input('P_el', val=np.ones((n,m)), desc='Electrical power output over time', units='W')
        self.add_output('P_in', val=np.ones(m), desc='Total power input over time', units='W')

        rows = np.arange(m).repeat(n)
        cols = np.arange(n*m).reshape((n,m)).flatten('F')
        self.declare_partials('P_in', 'P_el', rows=rows, cols=cols, val=1.)

    def compute(self,inputs,outputs):
        outputs['P_in'] = np.sum(inputs['P_el'], 0)

    def compute_partials(self, inputs, partials):
        pass

if __name__ == "__main__":
    
    #debug script:
        
    n_in = 5
    npts = 3

    prob = om.Problem()
    model = prob.model

    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    
    indeps.add_output('P_el', val=np.ones((n_in,npts)))
    
    model.add_subsystem('p', PowerInput(n_in=n_in, npts=npts), promotes=['*'])

    prob.setup(check=True)

    prob.run_model()

    print(prob['P_in'])

    check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=False, method='cs')