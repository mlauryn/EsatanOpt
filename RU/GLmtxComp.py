""" 
Component for generating thermal model linear conductor matrix based on input parameters and esatan data. 
This component takes thermal conductivities k as input variables and generates linear conductors GL = k * SF, 
where SF is shape factor given by SF = A/L (A-crossectional area of conductor, L - length of conductor). 
Initial GLs are to be provided from esatan model as option parameter GL_init
""" 
import openmdao.api as om
import numpy as np

class GLmtxComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n', types=int, desc='number of diffusion nodes in thermal model')
        self.options.declare('GL_init', desc='initial conductor matrix from thermal model as n+1 x n+1 array')
        self.options.declare('nodes', types=dict, desc='dictionary of node pair indice tuples (i,j) defining 2 nodes that each conductor connects')
        self.options.declare('SF', types=dict, desc='dictionary of shape factors for for each input conductor')
        # note: number of indices must be equal to number of conductor input variables with base 1
    
    def setup(self):    
        n = self.options['n'] + 1
        nodes = self.options['nodes']
        SF = self.options['SF']
        self.add_output('GL', shape=(n,n))
        for var in nodes:
            self.add_input(var) # adds input variable with the same name as user conductor name
            idx = nodes[var] # reads node pair indices of the input conductor
            self.declare_partials('GL', var, 
            rows=[(idx[0])*n+idx[0], (idx[0])*n+idx[1], (idx[1])*n+idx[0], (idx[1])*n+idx[1]],
            cols=[0,0,0,0],
            val=np.multiply([-1.,1.,1.,-1.], SF[var]))
        #self.declare_partials(of='GL', wrt='*', method='fd')
        # note: we define sparsity pattern of constant partial derivatives, openmdao expects shape (n*n, 1) 
    
    def compute(self, inputs, outputs):
        n = self.options['n'] + 1
        GL = np.copy(self.options['GL_init'])
        SF = self.options['SF']
        nodes = self.options['nodes'] 
        for var in inputs:
            idx = nodes[var]
            GL[idx] = SF[var]*inputs[var] # updates GL values based on input
        
        #GL = GL[1:,1:] # remove header row and column, as esatan base node numbering starts from 1

        #make GL matrix symetrical
        i_lower = np.tril_indices(n, -1)
        GL[i_lower] = GL.T[i_lower]

        #define diagonal elements as negative of all node conductor couplings (sinks)
        diag = np.negative(np.sum(GL, 1))

        di = np.diag_indices(n)
        GL[di] = diag

        GL[0,0] = 1.0 # deep space node temperature = 0 K
        
        outputs['GL'] = GL

    def compute_partials(self, inputs, partials):
        pass

if __name__ == "__main__":
    # script for testing partial derivs
    from Conductors import parse_cond
    from inits import inits
    n = 13
    nodes = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'
    GL_init, GR_init, QI_init, QS_init = inits(n, nodes, conductors)

    filepath = 'conductors.txt'
    data = parse_cond(filepath)
    nodes = {}
    shape_factors = {}
    values = {}
    for entry in data:
        nodes.update( {entry['cond_name'] : entry['nodes']} )
        shape_factors.update( {entry['cond_name'] : entry['SF'] } )
        values.update( {entry['cond_name'] : entry['conductivity'] } )  
    #print(shape_factors, nodes)

    model = om.Group()
    comp = om.IndepVarComp()
    for var in nodes:
        comp.add_output(var, val=values[var] ) # adds output variable with the same name as user conductor name
    
    
    model.add_subsystem('input', comp, promotes=['*'])
    model.add_subsystem('example', GLmtxComp(n=n, GL_init=GL_init, nodes=nodes, SF=shape_factors), promotes=['*'])

    """ model.connect('input.Spacer1', 'example.Spacer1')
    model.connect('input.Spacer2', 'example.Spacer2') """
    
    problem = om.Problem(model=model)
    problem.setup(check=True)
    
    problem.run_model()
    
    #check_partials_data = problem.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-02)

    print(problem['example.GL'])