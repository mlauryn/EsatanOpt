import openmdao.api as om
import numpy as np

class PowerOutput(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nn', types=int, desc='number of diffusion nodes in thermal model')
        self.options.declare('npts', default=1, types=int, desc='number of points')
    def setup(self):
        
        m = self.options['npts']
        nn = self.options['nn']+1
        
        self.add_input('QI', val=np.zeros((nn,m)), desc='Internal power dissipation over time for each node', units='W')
        self.add_output('P_out', val=np.ones(m), desc='Total power output over time', units='W')

    def compute(self,inputs,outputs):
        outputs['P_out'] = np.sum(inputs['QI'], 0)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        dP_out = d_outputs['P_out']

        if mode == 'fwd':
            dP_out += np.sum(d_inputs['QI'], 0)
        else:
            d_inputs['QI'] += dP_out

if __name__ == "__main__":
    
    #debug script:
        
    nn = 5
    npts = 3

    prob = om.Problem()
    model = prob.model

    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    
    indeps.add_output('QI', val=np.ones((nn+1,npts)))
    
    model.add_subsystem('p', PowerOutput(nn=nn, npts=npts), promotes=['*'])

    prob.setup(check=True)

    prob.run_model()

    print(prob['P_out'])

    check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-04)
        
    



