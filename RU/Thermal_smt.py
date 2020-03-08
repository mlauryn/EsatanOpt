import numpy as np
from smt.surrogate_models import KRG
import openmdao.api as om
import smt

# Remote unit thermal surrogate model with SMT Toolbox and OpenMDAO Explicit Comp 

class ThermoSurrogate(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('sm', types=smt.surrogate_models.krg.KRG)
        self.options.declare('case', values=['hot', 'cold'])
    def setup(self):        
        case = self.options['case']
        self.add_input('eps', val=0.02)
        self.add_input('length', val=0.1)
        self.add_input('eff', val=0.1)
        if case == 'cold':
            self.add_input('r_bat', val=0.5)
        self.add_input('R_m', val=0.4)
        self.add_input('R_p', val=0.4)
        self.add_input('R_s', val=0.4)
        self.add_output('tBat', val=0.0)
        self.add_output('tProp', val=0.0)
        self.add_output('tBPanel', val=0.0)
        self.add_output('tDPanel', val=0.0)
        self.declare_partials(of='*', wrt='*')
    def compute(self, inputs, outputs):
        sm = self.options['sm']
        x = np.column_stack([inputs[i] for i in inputs])        
        y = sm.predict_values(x) 
        for i,invar in enumerate(outputs):
            outputs[invar] = y[0,i]              
    def compute_partials(self, inputs, partials):   
        x = np.column_stack([inputs[i] for i in inputs])
        for i in inputs:
            dy_dx[i,:] = sm.predict_derivatives(x,i)
        for i,invar in enumerate(inputs):
            dy_dx = sm.predict_derivatives(x,i)
            for num,y in enumerate(outputs):
                partials[y, invar] = dy_dx[0,num]

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from smt.utils import compute_rms_error

    case = 'cold' # hot or cold
    
    if case == 'cold':
        ndim = 7
        train = np.loadtxt('./TrainingData/RUc_TrainingData[ese]_n=100.csv', delimiter=',')
        test = np.loadtxt('./TrainingData/RUc_TrainingData[ese]_n=50.csv', delimiter=',')
        xtest, ytest = test[:,:ndim], test[:,ndim]
        xt, yt = train[:,:ndim], train[:,ndim:]
    elif case == 'hot':
        ndim = 6
        train = np.loadtxt('./TrainingData/RUh_TrainingData[ese]_n=200.csv', delimiter=',')
        test = np.loadtxt('./TrainingData/RUh_TrainingData[ese]_n=100.csv', delimiter=',')
        xtest, ytest = test[:,:ndim], test[:,ndim]
        xt, yt = train[:,:ndim], train[:,ndim:]

    sm=KRG(theta0=[1e-2]*ndim,print_prediction = False)
    sm.set_training_values(xt, yt)
    sm.train()

    prob = om.Problem()
    model = prob.model

    # create and connect inputs and outputs
    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('length', val=0.1)
    indeps.add_output('eff', val=0.25)
    indeps.add_output('eps', val=0.1)
    indeps.add_output('R_m', val=250.)
    indeps.add_output('R_p', val=250)
    indeps.add_output('R_s', val=250)
    if case == 'cold':
        indeps.add_output('r_bat', val=0.8)
        indeps.add_output('ht_gain', val=1.0)
        indeps.add_output('q_s', val=150.)
    else:
        indeps.add_output('ht_gain', val=0.0)
        indeps.add_output('q_s', val=1365.) 

    model.add_subsystem('mm', ThermoSurrogate(sm=sm, case=case), promotes=['*'])

    from Esatan.RU_esatan import RU_esatan
    import os
    os.chdir('.\esatan') 
    model.add_subsystem('tm', RU_esatan(), promotes_inputs=['*'], promotes_outputs=[('tBat','tBat_real')])


    prob.setup(check=True)
    prob.run_model()

    print(prob['tBat'], prob['tBat_real'])

    # Prediction of the validation points
    y = sm.predict_values(xtest)
    print('Kriging,  err: '+ str(compute_rms_error(sm,xtest,ytest)))

"""     fig = plt.figure()
    plt.plot(ytest, ytest, '-', label='$y_{true}$')
    plt.plot(ytest, y[:,0], 'r.', label='$\hat{y}$')

    plt.xlabel('$y_{true}$')
    plt.ylabel('$\hat{y}$')

    plt.legend(loc='upper left')
    plt.title('Kriging model: validation of the prediction model')
    plt.show() """

"""     # Value of theta
    print("theta values",  t.optimal_theta) """


            
        

