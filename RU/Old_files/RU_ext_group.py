""" Code for running coupled thermal analysis with external code  """
import openmdao.api as om
from SolarCell import SolarCell
from RUextCodeComp import RUextCodeComp
import numpy as np


class RemoteUnitExt(om.Group):

    def setup(self):
        
        cycle = self.add_subsystem('cycle', om.Group(), promotes_inputs=['length', 'eps', 'r_bat', 'R_m', 'R_p', 'R_s' ], promotes_outputs=['*'])
        cycle.add_subsystem('sc', SolarCell(), promotes=['*'])
        cycle.add_subsystem('tm', RUextCodeComp(), promotes=['*'])
        # Nonlinear Block Gauss Seidel is a gradient free solver
        cycle.nonlinear_solver = om.NonlinearBlockGS()


if __name__ == '__main__':
    
    case = 'cold'
    model = RemoteUnit()

    param = om.IndepVarComp()
    param.add_output('length', val=0.2)
    param.add_output('eps', val=0.1)
    param.add_output('R_m', val=250)
    param.add_output('R_p', val=250)
    param.add_output('R_s', val=250)
    if case == 'cold':
        param.add_output('r_bat', val=0.8)
        param.add_output('q_s', val=150.)
        param.add_output('ht_gain', val=1.0)
    elif case == 'hot':
        param.add_output('q_s', val=1365.0)
        param.add_output('ht_gain', val=0.0)

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

    
    



