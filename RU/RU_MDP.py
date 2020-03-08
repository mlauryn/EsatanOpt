import openmdao.api as om
from RU_group import RemoteUnit as mda
import pickle
from openmdao.utils.mpi import MPI
from pprint import pprint

class RU_MDP(om.Group):

    def setup(self):
        
        # Create IndepVarComp for broadcast parameters.
        bp = om.IndepVarComp()
        bp.add_output('length', val=0.1)
        bp.add_output('eps', val=0.1)
        bp.add_output('R_m', val=250)
        bp.add_output('R_p', val=250)
        bp.add_output('R_s', val=250)
        bp.add_output('r_bat', val=0.8)
        self.add_subsystem('bp', bp, promotes=['*'])

        pt0 = mda('hot')
        pt1 = mda('cold')

        # Remote unit instances go into a Parallel Group
        para = self.add_subsystem('parallel', om.ParallelGroup(), promotes=['*'])
        para.add_subsystem('pt0', pt0, promotes_inputs=['*'])
        para.add_subsystem('pt1', pt1, promotes_inputs=['*'])


if __name__ == '__main__':

    # import pylab
    import time

    model = RU_MDP()

    model.add_design_var('length', lower = 0.0, upper=0.254)
    model.add_design_var('eps', lower = 0.02, upper=0.8)
    model.add_design_var('r_bat', lower = 0.0, upper=1.0)
    model.add_design_var('R_m', lower = 1., upper=250.0)
    model.add_design_var('R_p', lower = 1., upper=250.0)
    model.add_design_var('R_s', lower = 1., upper=250.0)

    #constraint for  temperatures
    model.add_constraint('pt0.tBat', lower=0.0, upper = 45.0)
    model.add_constraint('pt0.tProp', lower=-10.0, upper = 80.0)
    model.add_constraint('pt1.tBat', lower=0.0, upper = 45.0)
    model.add_constraint('pt1.tProp', lower=-10.0, upper = 80.0)
    #model.add_constraint('tMain', lower=-40.0, upper = 85.0)
    
    #model.add_subsystem('obj', ExecComp('length'), promotes=['*'])
    model.add_objective('length')

    # create problem and add optimizer
    prob = om.Problem(model)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer']='SLSQP'
    prob.driver.options['disp'] = True
    #prob.driver.opt_settings = {'eps': 1.0e-6, 'ftol':1e-04,} 
    prob.driver.add_recorder(om.SqliteRecorder("ru_mdp.sql"))
    
    prob.setup()

    t = time.time()
    prob.run_driver()
    print('time:', time.time() - t)
    # save result (objective and constraints) to a pickle
    
    npts = 2
    data = {'obj.val': prob['length'], 'eps': prob['eps'], 'R_m': prob['R_m'], 'R_p': prob['R_p'], 'R_s': prob['R_s'],
    'eff_h': prob['pt0.cycle.eff'], 'eff_c': prob['pt1.cycle.eff'], 'r_bat': prob['r_bat']}
    cons = ['pt%d.tBat', 'pt%d.tProp', 'pt%d.tBPanel', 'pt%d.tDPanel']
    

    for pt in range(npts):
        for con in [con % pt for con in cons]:
            data[con] = prob[con]
    pprint(data)
    pickle.dump(data, open('mdp.p', 'wb'))
