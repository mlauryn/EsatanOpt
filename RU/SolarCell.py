#Simple component that computes the dependancy of remote unit solar cell efficiency with temperature.

import openmdao.api as om
import numpy as np 

class SolarCell(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nodes', types=list, desc='list of input external surface node numbers')
        self.options.declare('npts', default=1, types=int, desc='number of points')  
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
        T0 = 28. #reference temperature
        eff0 = .285 #efficiency at ref temp
        T1 = -150.
        eff1 = 0.335

        delta_T = inputs['T'] - T0

        slope = (eff1 - eff0) / (T1 - T0)
        
        outputs['eta'] = eff0 + slope * delta_T

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        
        T0 = 28.
        eff0 = .285
        T1 = -150.
        eff1 = 0.335
        slope = (eff1 - eff0) / (T1 - T0)
        
        deff_dT = d_outputs['eta']

        if mode == 'fwd':
            
            deff_dT += slope * d_inputs['T']
        else:

            d_inputs['T'] = slope * deff_dT

if __name__ == "__main__":
    
    nodes=[2,4]
    m=2
    n = len(nodes)
    nn = 16

    prob = om.Problem()
    model = prob.model
    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('T', val=np.ones((nn,m))*273., units='K')
    
    model.add_subsystem('sc', SolarCell(npts=m, nodes=nodes), promotes=['*'])

    prob.setup(check=True)
    prob.run_model()

    print(prob['eta'])

    check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-03)



