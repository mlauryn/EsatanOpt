import openmdao.api as om
import numpy as np

class QSmtxComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nn', types=int, desc='number of diffusion nodes in thermal model')
        self.options.declare('nodes', types=list, desc='list of input external surface node numbers')
        self.options.declare('npts', default=1, types=int, desc='number of points')
        
    def setup(self):
        nn = self.options['nn'] + 1
        n = len(self.options['nodes'])
        m = self.options['npts']
        self.add_input('P_el', shape=(n,m), desc='solar cell electric power over time', units='W')
        self.add_input('QS_c', shape=(n,m), desc='solar cell absorbed heat over time', units='W')
        self.add_input('QS_r', shape=(n,m), desc='radiator absorbed heat over time', units='W')
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


if __name__ == "__main__":
    
    #debug script:  
    nn = 4
    npts = 2
    n = [1,4] #nodes

    prob = om.Problem()
    model = prob.model

    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('QS_c', val=np.ones((len(n),npts)))
    indeps.add_output('QS_r', val=np.ones((len(n),npts)))
    indeps.add_output('P_el', val=np.ones((len(n),npts)))

    
    model.add_subsystem('QS', QSmtxComp(nn=nn, nodes=n, npts=npts), promotes=['*'])

    prob.setup(check=True)

    prob.run_model()

    print(prob['QS'])

    check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-04)

    


