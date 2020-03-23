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
            deriv = area[i] * VF[i] * sigma
            self.declare_partials('GR', name, rows=[node, node * n + node], cols=[0, 0], val=[deriv, -1 * deriv ])
        
        # note: we define sparsity pattern of constant partial derivatives, openmdao expects shape (n*n, 1) 
    
    def compute(self, inputs, outputs):
        n = self.options['n'] + 1
        GR = np.copy(self.options['GR_init'])
        VF = self.options['VF']
        area = self.options['A']
        nodes = self.options['nodes']
        sigma = 5.670374e-8

        for i, eps in enumerate(inputs):
            GR[0, nodes[i]] = area[i] * VF[i] * inputs[eps] * sigma # updates REFs to deep space based on input emissivity, view factor and area
        
        #need to update diagonals
        
        di = np.diag_indices(n)
        GR[di] = np.zeros(n)
        diag = np.negative(np.sum(GR, 0))
        GR[di] = diag
        
        outputs['GR'] = GR

    def compute_partials(self, inputs, partials):
        pass

if __name__ == "__main__":
    # script for testing partial derivs
    from ViewFactors import parse_vf
    from inits import inits
    
    nodals = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'
    n, GL_init, GR_init, QI_init, QS_init = inits(nodals, conductors)

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
    

    print((problem['example.GR'][0,1:12] - GR_init[0,1:12])/GR_init[0,1:12])
    #print(problem['example.GR'] == GR_init)
    #problem.model.list_inputs(print_arrays=True)