import openmdao.api as om
from RU_group import RemoteUnit as mda
import pickle
from openmdao.utils.mpi import MPI
from pprint import pprint
import numpy as np

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

class penaltyFunction(om.ExplicitComponent):
    """
    Evaluates temperature constraint violation as 
    component actual and required temperature difference 
    if constraint is violated and 0 if not violated
    """
    def setup(self):
        self.add_input('tBat_c',val=0.0)
        self.add_input('tBat_h',val=0.0)
        self.add_input('tProp_c',val=0.0)
        self.add_input('tProp_h',val=0.0)
        #self.add_input('tMain_h',val=0.0)
        #self.add_input('tMain_c',val=0.0)
        #self.add_input('tTether_h',val=0.0)
        self.add_output('penalty',val=0.0)
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        tBat_c = inputs['tBat_c']
        tBat_h = inputs['tBat_h']
        tProp_c = inputs['tProp_c']
        tProp_h = inputs['tProp_h']
        #tMain_h = inputs['tMain_h']
        #tMain_c = inputs['tMain_c']
        #tTether_h = inputs['tTether_h']
        tBat_min = .0 
        tProp_min = -10.0 
        #tMain_min = -35.0 
        tBat_max = 45.0 
        tProp_max = 80.0 
        #tMain_max = 80.0
        #tTether_max = 75.0
        deltatBat_c = tBat_min - tBat_c 
        deltatProp_c = tProp_min - tProp_c 
        #deltatMain_c = tMain_min - tMain_c 
        deltatBat_h = tBat_h - tBat_max 
        deltatProp_h = tProp_h - tProp_max 
        #deltatMain_h = tMain_h - tMain_max 
        #deltatTether_h = tTether_h - tTether_max
        mar = 5.0 #temperature margin

        delta = np.array([deltatBat_c, deltatProp_c, deltatBat_h, deltatProp_h]) #temperature violations
        
        #weights = np.array([0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.1])
        norm_con = delta / mar #* weights #normalized contraints
        #con = np.power(norm_con, 2) 

        outputs['penalty'] = np.sum(norm_con)

        def compute_partials(self, inputs, partials):
            
            mar = 5.0 #temperature margin

            partials['penalty', 'tBat_c'] = - 1/mar
            partials['penalty', 'tProp_c'] = - 1/mar
            partials['penalty', 'tBat_h'] = 1/mar
            partials['penalty', 'tProp_h'] = 1/mar

if __name__ == '__main__':

    # import pylab
    import time

    model = RU_MDP()

    model.add_subsystem('con', penaltyFunction(), promotes=['*'])
    
    #objective function is panel length plus penalty of violating temperature constraints
    model.add_subsystem('obj', om.ExecComp('obj = length + penalty'), promotes=['*'])

    #connect different outputs to constraint violation inputs
    model.connect('pt0.tBat','tBat_h'), 
    #model.connect('pt0.tMain','tMain_h')
    model.connect('pt0.tProp','tProp_h')
    #model.connect('pt0.tTether','tTether_h')
    model.connect('pt1.tBat','tBat_c') 
    #model.connect('pt1.tMain','tMain_c')
    model.connect('pt1.tProp','tProp_c')

    model.add_design_var('length', lower = 0.0, upper=0.254)
    model.add_design_var('eps', lower = 0.02, upper=0.8)
    model.add_design_var('r_bat', lower = 0.0, upper=1.0)
    model.add_design_var('R_m', lower = 1., upper=250.0)
    model.add_design_var('R_p', lower = 1., upper=250.0)
    model.add_design_var('R_s', lower = 1., upper=250.0)

    """ #constraint for  temperatures
    model.add_constraint('pt0.tBat', lower=0.0, upper = 45.0)
    model.add_constraint('pt0.tProp', lower=-10.0, upper = 80.0)
    model.add_constraint('pt1.tBat', lower=0.0, upper = 45.0)
    model.add_constraint('pt1.tProp', lower=-10.0, upper = 80.0)
    #model.add_constraint('tMain', lower=-40.0, upper = 85.0) """
    
    model.add_objective('obj')
    #model.add_objective('length')

    # create problem and add optimizer
    prob = om.Problem(model)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer']='SLSQP'
    prob.driver.options['disp'] = True
    #prob.driver.options['maxiter'] = 500
    #prob.driver.opt_settings = {'eps': 1.0e-3, 'ftol':1e-04,} 
    prob.driver.add_recorder(om.SqliteRecorder("ru_mdp_2.sql"))
    
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
