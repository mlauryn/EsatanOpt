import openmdao.api as om
import numpy as np
import math

class Ilumination(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('G_sc', default=1365.0, desc='solar constant')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        self.options.declare('A', default=[1]*7, types=list, desc='input surface node areas')
    def setup(self):
        nn = len(self.options['A'])
        m = self.options['npts']
        self.add_input('dist', val=np.ones(m), desc='distane in AU')
        self.declare_partials(of='QIS', wrt='dist', dependent=False) # distance will be independat variable
        self.add_input('beta', val=np.ones(m), units='rad')
        self.add_output('QIS', val=np.ones((nn,m)), units='W')

    def compute(self, inputs, outputs):
        m = self.options['npts']
        nn = len(self.options['A'])
        d = inputs['dist']
        Gsc = self.options['G_sc']
        
        A = self.options['A']
        beta = inputs['beta']
        QIS = np.zeros((nn,m))

        q_s = Gsc * (1/d)**2 #solar flux at distance d

        for i in range(m):
            for nn in [0,2,3]: #these nodes are solar cells
                QIS[nn,i] = q_s[i] * A[nn] * math.cos(beta[i])
            QIS[6,i] = q_s[i] * A[6] * math.sin(beta[i])
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
                    for nn in [0,2,3]: #these nodes are solar cells
                        dQIS[nn,i] -= q_s[i] * A[nn] *  math.sin(beta[i])
                    dQIS[6,i] += q_s[i] * A[6] * math.cos(beta[i])
        else:
            
            if 'beta' in d_inputs:
                for i in range(m):
                    for nn in [0,2,3]: #these nodes are solar cells
                        d_inputs['beta'] -= q_s[i] * A[nn] * math.sin(beta[i]) * dQIS[nn,i]
                    d_inputs['beta'] += q_s[i] * A[6] * math.cos(beta[i]) * dQIS[6,i]

class SolarPower(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_in', types=int, desc='number of input nodes')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        #self.options.declare('eta_con', default=.95, lower=.0, upper=1., desc='MPPT converter efficiency')
        self.options.declare('alp_sc', default=.91, lower=.0, upper=1., desc='absorbtivity of the solar cell' )

    def setup(self):
        nn = self.options['n_in']
        m = self.options['npts']
        
        self.add_input('QIS', shape=(nn,m), desc='incident solar power', units='W')
        self.add_input('alp_r', shape=(nn,1), desc='absorbtivity of the input node radiating surface')
        self.add_input('cr', shape=(nn,1), desc='solar cell or radiator installation decision for input nodes')
        self.add_output('QS_c', shape=(nn,m), desc='solar cell absorbed power over time', units='W')
        self.add_output('QS_r', shape=(nn,m), desc='radiator absorbed power over time', units='W')

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

class ElectricPower(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_in', types=int, desc='number of input nodes')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        self.options.declare('ar', default=.90, lower=.0, upper=1., desc='solar cell to node surface area ratio')
        self.options.declare('eta_con', default=.95, lower=.0, upper=1., desc='MPPT converter efficiency')

    def setup(self):
        nn = self.options['n_in']
        m = self.options['npts']
    
        self.add_input('eta', val=np.ones((nn,m))*0.3/0.91, desc='solar cell efficiency with respect to absorbed power for input surface nodes over time ')
        self.add_input('QS_c', shape=(nn,m), desc='solar cell absorbed power over time', units='W')
        self.add_output('P_el', shape=(nn,m), desc='Electrical power output over time', units='W')
        #self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        ar = self.options['ar']
        m = self.options['npts']
        nn = self.options['n_in']
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

class QSmtxComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nn', types=int, desc='number of diffusion nodes in thermal model')
        self.options.declare('nodes', types=list, desc='list of input external surface node numbers')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        
    def setup(self):
        nn = self.options['nn'] + 1
        n_in = len(self.options['nodes'])
        m = self.options['npts']
        self.add_input('P_el', shape=(n_in,m), desc='solar cell electric power over time', units='W')
        self.add_input('QS_c', shape=(n_in,m), desc='solar cell absorbed heat over time', units='W')
        self.add_input('QS_r', shape=(n_in,m), desc='radiator absorbed heat over time', units='W')
        self.add_output('QS', val=np.zeros((nn,m)), desc='solar absorbed heat over time', units='W')
    
    def compute(self, inputs, outputs):
        nn = self.options['nn'] + 1
        m = self.options['npts']
        QS = np.zeros((nn,m))
        P_el = inputs['P_el']
        QS_c = inputs['QS_c']
        QS_r = inputs['QS_r']
        for i,node in enumerate(self.options['nodes']):
            QS[node,:] = QS_c[i,:] + QS_r[i,:] - P_el[i,:] # energy balance
        outputs['QS'] = QS
    
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        m = self.options['npts']
        nodes = self.options['nodes']
                
        P_el = inputs['P_el']
        QS_c = inputs['QS_c']
        QS_r = inputs['QS_r']

        dQS = d_outputs['QS']

        if mode == 'fwd':
            
            if 'P_el' in d_inputs:
                for i,node in enumerate(nodes):
                    dQS[node,:] -= d_inputs['P_el'][i,:]

            if 'QS_c' in d_inputs:
                for i,node in enumerate(nodes):
                    dQS[node,:] += d_inputs['QS_c'][i,:]
            
            if 'QS_r' in d_inputs:
                for i,node in enumerate(nodes):
                    dQS[node,:] += d_inputs['QS_r'][i,:]
        else:

            if 'P_el' in d_inputs:
                for i,node in enumerate(nodes):
                    d_inputs['P_el'][i,:] -= dQS[node,:]

            if 'QS_c' in d_inputs:
                for i,node in enumerate(nodes):
                    d_inputs['QS_c'][i,:] += dQS[node,:]
            
            if 'QS_r' in d_inputs:
                for i,node in enumerate(nodes):
                    d_inputs['QS_r'][i,:] += dQS[node,:]

class PowerInput(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_in', types=int, desc='number of input nodes')
        self.options.declare('npts', default=1, types=int, desc='number of points')
    def setup(self):
        nn = self.options['n_in']
        m = self.options['npts']
        self.add_input('P_el', val=np.ones((nn,m)), desc='Electrical power output over time', units='W')
        self.add_output('P_in', val=np.ones(m), desc='Total power input over time', units='W')
    def compute(self,inputs,outputs):
        outputs['P_in'] = np.sum(inputs['P_el'], 0)
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        dP_in = d_outputs['P_in']

        if mode == 'fwd':
            dP_in += np.sum(d_inputs['P_el'], 0)
        else:
            d_inputs['P_el'] += dP_in

class PowerOutput(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nn', types=int, desc='number of diffusion nodes in thermal model')
        self.options.declare('npts', default=1, types=int, desc='number of points')
    def setup(self):
        
        m = self.options['npts']
        nn = self.options['nn']+1
        
        self.add_input('QI', val=np.zeros((nn,npts)), desc='Internal power dissipation over time for each node', units='W')
        self.add_output('P_out', val=np.ones(npts), desc='Total power output over time', units='W')

    def compute(self,inputs,outputs):
        outputs['P_out'] = np.sum(inputs['QI'], 0)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        dP_out = d_outputs['P_out']

        if mode == 'fwd':
            dP_out += np.sum(d_inputs['QI'], 0)
        else:
            d_inputs['QI'] += dP_out

if __name__ == "__main__":

    from ViewFactors import parse_vf
    from inits import inits
    
    npts = 1
    nodals = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'
    nn, GL_init, GR_init, QI_init, QS_init = inits(nodals, conductors)

    view_factors = 'viewfactors.txt'
    data = parse_vf(view_factors)
    nodes = []
    area = []
    vf = []
    eps = []
    for entry in data:
        nodes.append(entry['node number'])
        area.append(entry['area'])
        vf.append(entry['vf']) 
        eps.append(entry['emissivity'])  
    #print(nodes, area, vf, eps)

    n_in = len(area)

    model = om.Group()
    params = om.IndepVarComp()
    params.add_output('beta', val=np.zeros(npts) )
    params.add_output('dist', val=np.ones(npts)*3.)
    params.add_output('QI', val=QI_init)
    
    model.add_subsystem('params', params, promotes=['*'])
    model.add_subsystem('il', Ilumination(npts=npts, A=area), promotes=['*'])
    model.add_subsystem('sol', SolarPower(n_in=n_in, npts=npts), promotes=['*'])
    model.add_subsystem('el', ElectricPower(n_in=n_in, npts=npts), promotes=['*'])
    model.add_subsystem('QS', QSmtxComp(nn=nn, nodes=nodes, npts=npts), promotes=['*'])
    model.add_subsystem('Pin', PowerInput(n_in=n_in, npts=npts), promotes=['*'])
    model.add_subsystem('Pout', PowerOutput(nn=nn, npts=npts), promotes=['*'])

    
    problem = om.Problem(model=model)
    problem.setup(check=True)
    
    problem.run_model()
    
    #check_partials_data = problem.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-02)
    
    print(problem['P_in'])
    print(problem['P_out'])
    print(QS_init)
    
    print(problem['QS'])
    #problem.model.list_inputs(print_arrays=True)