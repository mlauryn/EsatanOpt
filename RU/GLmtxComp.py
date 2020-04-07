""" 
Component for generating thermal model linear conductor matrix based on input parameters and esatan conductors. 
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
        self.options.declare('user_links', types=list, desc='list of user conductor data dictionaries')
    
    def setup(self):    
        n = self.options['n'] + 1
        conductors = self.options['user_links']       
        self.add_output('GL', shape=(n,n), units='W/K')
               
        for invar in conductors:
            name = invar['cond_name']
            nodes = invar['nodes']
            shape_factors = invar['SF']
            self.add_input(name) # adds input conductivity (scalar) with the same name as user conductor name

            partials = np.zeros((n,n))

            for idx, SF in zip(nodes, shape_factors):
                partials[idx] = SF  # derivative of dGL/dk = SF
            # do the same as in compute function
            i_lower = np.tril_indices(n, -1)
            partials[i_lower] = partials.T[i_lower]
            di = np.diag_indices(n)
            diag = np.negative(np.sum(partials, 1))
            partials[di] = diag
            
            # note: we define sparsity pattern of constant partial derivatives, openmdao expects shape (n*n, 1)
            flat_partials = partials.flatten()
            rows = np.nonzero(flat_partials)[0]
            values = flat_partials[rows]
            self.declare_partials('GL', name, rows=rows, cols=[0]*len(rows), val=values)
    
    def compute(self, inputs, outputs):
        n = self.options['n'] + 1
        GL = np.copy(self.options['GL_init'])
        conductors = self.options['user_links']
         
        for invar in conductors:
            name = invar['cond_name']
            nodes = invar['nodes']
            shape_factors = invar['SF']
            for idx, SF in zip(nodes, shape_factors):
                GL[idx] = SF * inputs[name]  # updates GL values based on input

        # mirror values from upper triangle to lower triangle
        i_lower = np.tril_indices(n, -1)
        GL[i_lower] = GL.T[i_lower]

        #define diagonal elements as negative of all node conductor couplings (sinks)
        
        di = np.diag_indices(n)
        GL[di] = np.zeros(n) # delete old result
        diag = np.negative(np.sum(GL, 1))
        GL[di] = diag

        GL[0,0] = 1.0 # deep space node temperature = 0 K (this coef is needed to avoid singularity in heat equations)
        
        outputs['GL'] = GL

    def compute_partials(self, inputs, partials):
        pass

if __name__ == "__main__":
    # script for testing partial derivs
    from Pre_process import parse_cond, nodes, conductors
    
    node_data = 'nodal_data.csv'
    cond_data = 'Cond_data.csv'
    nn, groups = nodes(node_data)
    GL_init, GR_init = conductors(nn, cond_data) 

    filepath = 'conductors.txt'
    conductors = parse_cond(filepath)

    model = om.Group()
    comp = om.IndepVarComp()
    for cond in conductors:
        comp.add_output(cond['cond_name'], val=cond['values'][0] ) # adds output variable with the same name as user conductor name
    
    
    model.add_subsystem('input', comp, promotes=['*'])
    model.add_subsystem('example', GLmtxComp(n=nn, GL_init=GL_init, user_links=conductors), promotes=['*'])

    """ model.connect('input.Spacer1', 'example.Spacer1')
    model.connect('input.Spacer2', 'example.Spacer2') """
    
    problem = om.Problem(model=model)
    problem.setup(check=True)
    
    problem.run_model()
    
    check_partials_data = problem.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-02)

    #print(problem['example.GL'])

    import sys

    np.set_printoptions(threshold=sys.maxsize)

    #print(problem['example.GL'] - GL_init)
    #print(problem['example.GL'])