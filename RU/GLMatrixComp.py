""" Component for generating thermal model linear conductor matrix based on input parameters and esatan data """ 
import openmdao.api as om
import numpy as np

class GLmtxComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n', types=int, desc='number of diffusion nodes in thermal model')
        self.options.declare('GL_init', desc='initial conductor matrix from thermal model as n x n array')
        self.options.declare('indices', types=list, desc='list of indices (i,j) for each input conductor defining 2 nodes that it connects')
        # note: number of indices must be equal to number of conductor input variables with base 1
    
    def setup(self):    
        self.add_output('GL', shape=(n,n))
        for idx in self.options['indices']:
            i_name = 'GL{}'.format(idx)
            self.add_input(i_name, units='W/K')
            self.declare_partials('GL', i_name, 
            rows=[(idx[0]-1)*n+idx[0]-1, (idx[0]-1)*n+idx[1]-1, (idx[1]-1)*n+idx[0]-1, (idx[1]-1)*n+idx[1]-1],
            cols=[0,0,0,0],
            val=[-1.,1.,1.,-1.])
        # note: we define sparsity pattern of constant partial derivatives, openmdao expects shape (n*n, 1) 
    
    def compute(self, inputs, outputs):
        GL = self.options['GL_init']
        for idx, invar in zip(self.options['indices'], inputs):
            GL[idx] = inputs[invar]
            #remove header row and column, as esatan base node numbering starts from 1
        GL = GL[1:,1:]
        outputs['GL'] = GL

    def compute_partials(self, inputs, partials):
        pass
        
        """ n = self.options['n']+1 #we use a base of 1 for simplicity
        for idx, invar in zip(self.options['indices'], inputs):
            d_GL = np.zeros((n,n))
            d_GL[idx] = 1.0 #derivative wrt itself is one
            d_GL[idx[1],idx[1]] = -1.0 # this is a diagonal contribution
            i_lower = np.tril_indices(n, -1)
            d_GL[i_lower] = d_GL.T[i_lower] #mirror matrix around diagonal
            d_GL = d_GL[1:,1:] #remove header row and column, as esatan base starts from 1
            partials['GL', invar] = d_GL """

if __name__ == "__main__":
    model = om.Group()
    comp = om.IndepVarComp()
    comp.add_output('GL(1, 2)')
    comp.add_output('GL(1, 13)')
    comp.add_output('GL(2, 3)')
    
    n = 13
    GL_init = np.zeros((n+1,n+1))
    indices = [(1, 2), (1, 13), (2, 3)]
    
    model.add_subsystem('input', comp)
    model.add_subsystem('example', GLmtxComp(n=n, GL_init=GL_init, indices=indices))

    model.connect('input.GL(1, 2)', 'example.GL(1, 2)')
    model.connect('input.GL(1, 13)', 'example.GL(1, 13)')
    model.connect('input.GL(2, 3)', 'example.GL(2, 3)')

    problem = om.Problem(model=model)
    problem.setup()
    problem.run_model()
    totals = problem.compute_totals(['example.GL'], ['input.GL(2, 3)', 'input.GL(1, 13)', 'input.GL(1, 2)'])

    print(np.reshape(totals['example.GL', 'input.GL(1, 13)'], (n,n)))