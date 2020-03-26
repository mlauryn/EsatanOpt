import openmdao.api as om
import numpy as np

class ElectricPower(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_in', types=int, desc='number of input nodes')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        self.options.declare('ar', default=.90, lower=.0, upper=1., desc='solar cell to node surface area ratio')
        self.options.declare('eta_con', default=.95, lower=.0, upper=1., desc='MPPT converter efficiency')

    def setup(self):
        n = self.options['n_in']
        m = self.options['npts']
    
        self.add_input('eta', val=np.ones((n,m))*0.3/0.91, desc='solar cell efficiency with respect to absorbed power for input surface nodes over time ')
        self.add_input('QS_c', shape=(n,m), desc='solar cell absorbed power over time', units='W')
        self.add_output('P_el', shape=(n,m), desc='Electrical power output over time', units='W')
        #self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        ar = self.options['ar']
        m = self.options['npts']
        n = self.options['n_in']
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


if __name__ == "__main__":
    #debug script:
        
    n_in = 5
    npts = 3

    prob = om.Problem()
    model = prob.model

    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    #indeps.add_output('eta', val=np.ones((n_in, npts))*0.3)
    indeps.add_output('QS_c', val=np.ones((n_in,npts))*3.0)
    indeps.add_output('eta', val=np.ones((n_in,npts))*0.5,)
    
    model.add_subsystem('p', ElectricPower(n_in=n_in, npts=npts), promotes=['*'])

    prob.setup(check=True)

    prob.run_model()

    print(prob['P_el'])

    check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-04)