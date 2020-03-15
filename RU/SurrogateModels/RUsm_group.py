import openmdao.api as om
from SolarCell import SolarCell
from Thermal_smt import ThermoSurrogate
import numpy as np
from smt.surrogate_models import KRG

class RemoteUnit(om.Group):

    def __init__(self, case):
        super(RemoteUnit, self).__init__()

        self.case = case

    def setup(self):
        
        case = self.case

        #load training data for surrogate

        if case == 'cold':
            ndim = 7
            train = np.loadtxt('./TrainingData/RUc_TrainingData[ese]_n=100.csv', delimiter=',')
            xt, yt = train[:,:ndim], train[:,ndim:]
        elif case == 'hot':
            ndim = 6
            train = np.loadtxt('./TrainingData/RUh_TrainingData[ese]_n=100.csv', delimiter=',')
            xt, yt = train[:,:ndim], train[:,ndim:]

        #train surrogate and pass to model
        sm = KRG(theta0=[1e-2]*ndim,print_prediction = False)
        sm.set_training_values(xt, yt)
        sm.train()

        cycle = self.add_subsystem('cycle', om.Group(), promotes_inputs=['*'], promotes_outputs=['tBat', 'tProp', 'tBPanel', 'tDPanel'])
        cycle.add_subsystem('sc', SolarCell(), promotes_inputs=['tBPanel', 'tDPanel'], promotes_outputs=['eff'])
        cycle.add_subsystem('tm', ThermoSurrogate(sm=sm, case=case), promotes=['*'])
        # Nonlinear Block Gauss Seidel is a gradient free solver
        cycle.nonlinear_solver = om.NonlinearBlockGS()


if __name__ == '__main__':
    
    case = 'cold'
    model = RemoteUnit(case)

    param = om.IndepVarComp()
    param.add_output('length', val=0.2)
    #param.add_output('eff', val=0.25)
    param.add_output('eps', val=0.1)
    param.add_output('R_m', val=250)
    param.add_output('R_p', val=250)
    param.add_output('R_s', val=250)
    if case == 'cold':
        param.add_output('r_bat', val=0.8)
    model.add_subsystem('param', param, promotes=['*'])

    prob = om.Problem(model)
    prob.setup(check=True)
    prob.run_model()

    s = f"""
    {'-'*40}
    # Output:
    # tBat : {prob['tBat']}
    # tProp : {prob['tProp']}
    # tBPanel : {prob['tBPanel']}
    # tDPanel : {prob['tDPanel']}
    # eff : {prob['eff']}

    {'-'*40}
    """
    print(s)

    
    



