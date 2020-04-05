import numpy as np
from smt.surrogate_models import RMTB, RMTC
import openmdao.api as om
import smt
import pandas as pd
from parse_hf import parse_hf
 

class HeatFluxComp(om.ExplicitComponent):
    """
    Vectorized surrogate model to predict solar heat flux based on yaw angle
    Regularized minimal-energy tensor-product splines (RMTS) from SMT toolbox

    """
    def __init__(self, faces, npts, train_data=None, method='RMTB'):
        super(HeatFluxComp, self).__init__()

        self.npts = npts # number of points (phi angles) at which to evaluate results
        self.faces = faces # external faces that need to be evaluated
        nodes = [] #assemble node numbers of the faces
        for face in faces:
            nodes.extend(face['nodes'])

        self.ny = len(nodes) # number of outputs

        if not train_data:
            train_data = 'radiative_results.txt'
        df = parse_hf(train_data)

        data = df.loc[0].pivot(index='var', columns='node', values='IS')
        data = data.filter(items=nodes)  

        xt = data.index.to_numpy()
        yt = data.to_numpy()

        xlimits = np.array([[xt[0], xt[-1]]])

        if method == 'RMTB':
            
            self.sm = RMTB(
                xlimits=xlimits,
                energy_weight=1e-5,
                regularization_weight=1e-14,
                min_energy=True,
                num_ctrl_pts=50
            )
        else:
            self.sm = RMTC(
                xlimits=xlimits,
                num_elements=15,
                energy_weight=1e-5,
                regularization_weight=1e-14,
                min_energy=True
            )
        
        self.sm.set_training_values(xt, yt)
        self.sm.train()

    def setup(self):
        
        m = self.npts
        ny = self.ny
        self.add_input('phi', val=np.zeros(m), units='deg' )
        self.add_output('q_s', val=np.zeros((m,ny)))
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs): 

        outputs['q_s'] = self.sm.predict_values(inputs['phi'])   
          
    def compute_partials(self, inputs, partials):

        dy_dx = self.sm.predict_derivatives(inputs['phi'], 0)
        
        #need to fit into proper shape
        jac = np.zeros((m,n*m))
        n=np.shape(dy_dx)[1]
        m=np.shape(dy_dx)[0] 
        idx = np.array([i for i in range(n)])
        
        for i in range(m):
            jac[i,[idx+n*i]] = dy_dx[i,:]

        partials['q_s', 'phi'] = jac.T

if __name__ == "__main__":

    # test partials

    from ViewFactors import parse_vf
    from opticals import opticals
    from inits import nodes

    nn, groups = nodes()

    optprop = parse_vf('viewfactors.txt')

    keys = ['Box:outer', 'Panel_outer:back']
    faces = opticals(groups, keys, optprop)

    prob = om.Problem()
    model = prob.model

    # create and connect inputs and outputs
    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('phi', val=np.array([45.0, 45.]))

    model.add_subsystem('mm', HeatFluxComp(faces=faces, npts=2), promotes=['*'])

    prob.setup(check=True)
    prob.run_model()

    print(prob['q_s'])

    check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=False, form='central', step=1e-03)