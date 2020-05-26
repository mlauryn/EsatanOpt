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
        self.options.declare('faces', types=list, desc='names and optical properties of input faces')
    
    def setup(self):    
        n = self.options['n'] + 1
        faces = self.options['faces']
        #VF = self.options['VF']
        #area = self.options['A']
        sigma = 5.670374e-8
        self.add_output('GR', shape=(n,n))
        for face in faces:
            self.add_input(face['name']) # adds input variable as face node group label
            
            rows = []
            derivs = []
            for i,node in enumerate(face['nodes']):
                deriv = face['areas'][i] * face['VFs'][i] * sigma # derivative of REF with respect to this node emissivity
                rows.extend([node, node * n + node])
                derivs.extend([deriv, -1 * deriv ])
            
            self.declare_partials('GR', face['name'], rows=rows, cols=[0]*len(face['nodes'])*2, val=derivs)
        
        # note: we define sparsity pattern of constant partial derivatives, openmdao expects shape (n*n, 1) 
    
    def compute(self, inputs, outputs):
        n = self.options['n'] + 1
        GR = np.copy(self.options['GR_init'])
        faces = self.options['faces']
        sigma = 5.670374e-8

        for face in faces:
            GR[0, face['nodes']] = np.array(face['areas']) * np.array(face['VFs']) * sigma * inputs[face['name']] # updates REFs to deep space based on input emissivity, view factor and area
        
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
    from Pre_process import parse_vf, inits, nodes, opticals, conductors

    nn, groups = nodes('./Esatan_models/RU_v4_base/nodes_output.csv')
    GL_init, GR_init = conductors(nn, './Esatan_models/RU_v4_base/cond_output.csv') 

    optprop = parse_vf('./Esatan_models/RU_v4_base/vf_report.txt')

    #groups.update({'my_group': [54,55,56,57]})
    keys = ['Box', 'Panel_outer:back', 'Panel_inner:back'] #, 'my_group']
    faces = opticals(groups, keys, optprop)

    model = om.Group()
    params = om.IndepVarComp()
    for face in faces:
        name = face['name']
        value = face['eps'][0]
        params.add_output(name, val=value ) # adds independant variable as face name and assigns emissivity of it's first node
    
    model.add_subsystem('params', params, promotes=['*'])
    model.add_subsystem('example', GRmtxComp(n=nn, GR_init=GR_init, faces=faces), promotes=['*'])
    
    problem = om.Problem(model=model)
    problem.setup(check=True)
    
    problem.run_model()
    
    #check_partials_data = problem.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-02)
    

    #print((problem['example.GR'][0,1:nn] - GR_init[0,1:nn])/GR_init[0,1:nn])
    #print(problem['example.GR'] == GR_init)
    #problem.model.list_inputs(print_arrays=True)
    print(GR_init)