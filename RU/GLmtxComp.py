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
        self.options.declare('GL_init', desc='initial conductor matrix from thermal model as n x n array')
        self.options.declare('nodes', types=dict, desc='dictionary of input conductor names and indices (var,j) defining 2 nodes that it connects')
        self.options.declare('SF', types=dict, desc='dictionary of shape factors for for each input conductor')
        # note: number of indices must be equal to number of conductor input variables with base 1
    
    def setup(self):    
        n = self.options['n']
        nodes = self.options['nodes']
        SF = self.options['SF']
        self.add_output('GL', shape=(n,n))
        for var in nodes:
            self.add_input(var) # adds input variable with the same name as user conductor name
            idx = nodes[var] # takes the indices of the input conductor
            """ self.declare_partials('GL', var, 
            rows=[(idx[0]-1)*n+idx[0]-1, (idx[0]-1)*n+idx[1]-1, (idx[1]-1)*n+idx[0]-1, (idx[1]-1)*n+idx[1]-1],
            cols=[0,0,0,0],
            val=np.multiply([-1.,1.,1.,-1.], SF[var])) """
        #self.declare_partials(of='GL', wrt='*', method='fd')
        # note: we define sparsity pattern of constant partial derivatives, openmdao expects shape (n*n, 1) 
    
    def compute(self, inputs, outputs):
        n = self.options['n']
        GL = self.options['GL_init']
        SF = self.options['SF']
        nodes = self.options['nodes'] 
        for var in inputs:
            idx = nodes[var]
            GL[idx] = SF[var]*inputs[var] # updates GL values based on input

        GL = GL[1:,1:] # remove header row and column, as esatan base node numbering starts from 1

        #make GL matrix symetrical
        i_lower = np.tril_indices(n, -1)
        GL[i_lower] = GL.T[i_lower]

        #define diagonal elements as negative of all node conductor couplings (sinks)
        diag = -np.sum(GL, 1)

        di = np.diag_indices(n)
        GL[di] = diag
        
        outputs['GL'] = GL

    """ def compute_partials(self, inputs, partials):
        pass """

if __name__ == "__main__":
    from Conductors import _parse_line, parse_cond
    from inits import inits
    n = 13
    env = '99999'
    inact = '99998'
    nodes = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'
    GL_init, GR_init, QI_init, QS_init = inits(n, env, inact, nodes, conductors)

    filepath = 'conductors.txt'
    data = parse_cond(filepath)
    nodes = {}
    shape_factors = {}
    values = {}
    for entry in data:
        nodes.update( {entry['cond_name'] : tuple(map(int, entry['nodes'].split(',')))} )
        shape_factors.update( {entry['cond_name'] : float(entry['SF']) } )
        values.update( {entry['cond_name'] : float(entry['conductivity']) } )  
    
    model = om.Group()
    comp = om.IndepVarComp()
    for var in nodes:
        comp.add_output(var, val=values[var] ) # adds output variable with the same name as user conductor name
    
    
    model.add_subsystem('input', comp, promotes=['*'])
    model.add_subsystem('example', GLmtxComp(n=n, GL_init=GL_init, nodes=nodes, SF=shape_factors), promotes=['*'])

    """ model.connect('input.Spacer1', 'example.Spacer1')
    model.connect('input.Spacer2', 'example.Spacer2') """
    GL_init = GL_init[1:,1:]

    #make GL_init matrix symetrical
    i_lower = np.tril_indices(n, -1)
    GL_init[i_lower] = GL_init.T[i_lower]

    #define diagonal elements as negative of all node conductor couplings (sinks)
    diag = -np.sum(GL_init, 1)

    di = np.diag_indices(n)
    GL_init[di] = diag

    problem = om.Problem(model=model)
    problem.setup(check=True)
    problem.run_model()
    
    """ data = problem.check_partials()

    x1_error = data['example']['GL', 'Spacer1']['abs error']

    print(x1_error.forward)
    print(x1_error.reverse)

    x2_error = data['example']['GL', 'Spacer2']['abs error']

    print(x2_error.forward)
    print(x2_error.reverse) """


    totals = problem.compute_totals(['GL'], ['Spacer2', 'Spacer1'])
    #print(np.reshape(totals['example.GL', 'input.Spacer2'], (n,n)))
    print(problem['example.GL'])