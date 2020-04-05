import openmdao.api as om
import numpy as np

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

if __name__ == "__main__":
    #debug script:
    
    from ViewFactors import parse_vf
    from opticals import opticals
    from inits import nodes

    nn, groups = nodes()

    optprop = parse_vf('viewfactors.txt')

    keys = ['Box:outer', 'Panel_outer:back']
    faces = opticals(groups, keys, optprop)    
    
    nodes = []
    for face in faces:
        nodes.extend(face['nodes'])
    n_in = len(nodes)

    prob = om.Problem()
    model = prob.model

    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('q_s', val=np.ones((2,n_in))*1000.)
    
    model.add_subsystem('QIS', Incident_Solar(n_in=n_in, npts=2, faces=faces), promotes=['*'])

    prob.setup(check=True)

    prob.run_model()

    check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-04)

    print(prob['QIS'])