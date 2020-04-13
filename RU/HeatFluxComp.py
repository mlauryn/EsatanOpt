import numpy as np
from smt.surrogate_models import RMTB, RMTC
import openmdao.api as om
import smt
from Pre_process import parse_hf
 
class HeatFluxComp(om.ExplicitComponent):
    """
    Vectorized surrogate model to predict solar heat flux based on yaw angle
    Regularized minimal-energy tensor-product splines (RMTS) from SMT toolbox

    """
    def __init__(self, faces, npts, model=None, method='RMTB'):
        super(HeatFluxComp, self).__init__()

        self.npts = npts # number of points (phi angles) at which to evaluate results
        self.faces = faces # external faces that need to be evaluated
        nodes = [] #assemble node numbers of the faces
        for face in faces:
            nodes.extend(face['nodes'])

        self.ny = len(nodes) # number of outputs

        if not model:
            data = 'sdf_RU_v4_detail.txt'
        else:
            data = 'sdf_' + model + '.txt'
        train_data = parse_hf(data)

        xt = train_data[:,0]
        yt = train_data[:,nodes]

        xlimits = np.array([[xt[0], xt[-1]]])

        if method == 'RMTB':
            
            self.sm = RMTB(
                print_prediction=False,
                print_solver=False,
                xlimits=xlimits,
                energy_weight=1e-5,
                regularization_weight=1e-14,
                min_energy=True,
                num_ctrl_pts=50
            )
        else:
            self.sm = RMTC(
                print_prediction=False,
                print_solver=False,
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

        y = self.sm.predict_values(inputs['phi'])

        outputs['q_s'] = np.absolute(y)  # sometimes surrogate predicts negative values that are close to zero, so need to take absolute here
          
    def compute_partials(self, inputs, partials):

        dy_dx = self.sm.predict_derivatives(inputs['phi'], 0)
        
        #need to fit into proper shape
        n = np.shape(dy_dx)[1]
        m = np.shape(dy_dx)[0] 
        jac = np.zeros((m,n*m))
        idx = np.array([i for i in range(n)])
        
        for i in range(m):
            jac[i,[idx+n*i]] = dy_dx[i,:]

        partials['q_s', 'phi'] = jac.T

if __name__ == "__main__":

    # test partials
    from Pre_process import parse_vf, opticals, nodes

    nn, groups = nodes(data='nodes_RU_v4_base_cc.csv')

    optprop = parse_vf('vf_RU_v4_base.txt')

    keys = ['Box', 'Panel_outer']
    faces = opticals(groups, keys, optprop)
    
    prob = om.Problem()
    model = prob.model

    # create and connect inputs and outputs
    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('phi', val=np.array([45., 30.]))

    model.add_subsystem('mm', HeatFluxComp(faces=faces, npts=2, model='RU_v4_base'), promotes=['*'])

    prob.setup(check=True)
    prob.run_model()

    print(prob['q_s'])

    check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=False, form='forward', step=1e-04)