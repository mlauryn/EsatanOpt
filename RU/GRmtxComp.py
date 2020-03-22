""" 
Component for assembling thermal model radiative exchange factor (REF) matrix based on input parameters and esatan model data. 
REFs to deep space for each external surface node are calculated by GR = A * vf * eps * sigma, where A - area, eps - IR emissivity, vf - view factor to deep space. 
Emissivity is taken as input parameter. Remaining REFs are imported from the model as option parameter GR_init
""" 
import openmdao.api as om
import numpy as np

class GRmtxComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n', types=int, desc='number of diffusion nodes in thermal model')
        self.options.declare('GR_init', desc='initial REF matrix from thermal model as n+1 x n+1 array')
        self.options.declare('nodes', types=list, desc='list of input node numbers')
        self.options.declare('VF', types=list, desc='list of input node view factors')
        self.options.declare('A', types=list, desc='list of input node areas')
    
    def setup(self):    
        n = self.options['n'] + 1
        nodes = self.options['nodes']
        VF = self.options['VF']
        area = self.options['A']
        sigma = 5.670374e-8
        self.add_output('GR', shape=(n,n))
        for i,node in enumerate(nodes):
            name = 'eps:{}'.format(node)
            self.add_input(name) # adds input variable as 'emissivity:node no.'
            self.declare_partials('GR', name, rows=[node*n], cols=[0], val=[area[i] * VF[i] * sigma])
        
        # note: we define sparsity pattern of constant partial derivatives, openmdao expects shape (n*n, 1) 
    
    def compute(self, inputs, outputs):
        n = self.options['n'] + 1
        GR = np.copy(self.options['GR_init'])
        VF = self.options['VF']
        area = self.options['A']
        nodes = self.options['nodes']
        sigma = 5.670374e-8

        for i, invar in enumerate(inputs):
            GR[nodes[i],0] = area[i] * VF[i] * inputs[invar] * sigma # updates GR values based on input parameters
        
        outputs['GR'] = GR

    def compute_partials(self, inputs, partials):
        pass

if __name__ == "__main__":
    # script for testing partial derivs
    from ViewFactors import parse_vf
    from inits import inits
    
    n = 13
    
    GR_init = np.zeros((n+1,n+1))

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

    model = om.Group()
    params = om.IndepVarComp()
    for i,node in enumerate(nodes):
        name = 'eps:{}'.format(node)
        params.add_output(name, val=eps[i] ) # adds output variable as 'emissivity:node no.'
    
    model.add_subsystem('params', params, promotes=['*'])
    model.add_subsystem('example', GRmtxComp(n=n, GR_init=GR_init, nodes=nodes, VF=vf, A=area), promotes=['*'])
    
    problem = om.Problem(model=model)
    problem.setup(check=True)
    
    problem.run_model()
    
    check_partials_data = problem.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-02)

    #print(problem['example.GR'])