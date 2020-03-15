#Python script for genetic optimization of MAT remote unit thermal model as external code 
import time, math
import numpy as np
import openmdao.api as om
from RU_group import RemoteUnit as mda
import pickle
from pprint import pprint 


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
        self.add_input('tMain_h',val=0.0)
        self.add_input('tMain_c',val=0.0)
        self.add_input('tTether_h',val=0.0)
        self.add_output('penalty',val=0.0)

    def compute(self, inputs, outputs):
        tBat_c = inputs['tBat_c']
        tBat_h = inputs['tBat_h']
        tProp_c = inputs['tProp_c']
        tProp_h = inputs['tProp_h']
        tMain_h = inputs['tMain_h']
        tMain_c = inputs['tMain_c']
        tTether_h = inputs['tTether_h']
        tBat_min = .0 
        tProp_min = -10.0 
        tMain_min = -40.0 
        tBat_max = 45.0 
        tProp_max = 80.0 
        tMain_max = 85.0
        tTether_max = 80.0
        deltatBat_c = tBat_min - tBat_c 
        deltatProp_c = tProp_min - tProp_c 
        deltatMain_c = tMain_min - tMain_c 
        deltatBat_h = tBat_h - tBat_max 
        deltatProp_h = tProp_h - tProp_max 
        deltatMain_h = tMain_h - tMain_max 
        deltatTether_h = tTether_h - tTether_max
        mar = 5.0 #temperature margin

        delta = np.array([max(0, deltatBat_c), max(0, deltatProp_c), \
            max(0, deltatMain_c), max(0, deltatBat_h), max(0, deltatProp_h), \
            max(0, deltatMain_h), max(0, deltatTether_h)]) #temperature violations
        
        weights = np.array([0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.1])
        norm_con = delta / mar * weights #normalized contraints
        con = np.power(norm_con, 2) 

        outputs['penalty'] = np.sum(con)

prob = om.Problem()
model = prob.model
 
bp = om.IndepVarComp()
bp.add_output('length', val=0.2)
bp.add_output('eps', val=0.1)
bp.add_output('R_m', val=250)
bp.add_output('R_p', val=250)
bp.add_output('R_s', val=250)
bp.add_output('r_bat', val=0.5)
bp.add_output('qs_0', val=1365.)
bp.add_output('qs_1', val=150.)
bp.add_output('htgain_0', val=0.0)
bp.add_output('htgain_1', val=1.0)
model.add_subsystem('bp', bp, promotes=['*'])

#instantiate remote unit mda groups
pt0 = mda()
pt1 = mda()

# Remote unit mda groups go into a multipoint design Group
para = model.add_subsystem('mdp', om.Group(), promotes=['*'])
para.add_subsystem('pt0', pt0, promotes_inputs=['length', 'eps', 'r_bat', 'R_m', 'R_p', 'R_s' ])
para.add_subsystem('pt1', pt1, promotes_inputs=['length', 'eps', 'r_bat', 'R_m', 'R_p', 'R_s' ])
model.add_subsystem('con', penaltyFunction(), promotes=['*'])

#objective function is panel length plus penalty of violating temperature constraints
model.add_subsystem('obj', om.ExecComp('obj_p = length + penalty'), promotes=['*'])

#broadcast different boundary conditions for hot and cold analysis cases
model.connect('qs_0', 'pt0.cycle.q_s')
model.connect('qs_1', 'pt1.cycle.q_s')
model.connect('htgain_0', 'pt0.cycle.ht_gain')
model.connect('htgain_1', 'pt1.cycle.ht_gain')

#connect different outputs to constraint violation inputs
model.connect('pt0.tBat','tBat_h'), 
model.connect('pt0.tMain','tMain_h')
model.connect('pt0.tProp','tProp_h')
model.connect('pt0.tTether','tTether_h')
model.connect('pt1.tBat','tBat_c') 
model.connect('pt1.tMain','tMain_c')
model.connect('pt1.tProp','tProp_c')

""" #constraint for  temperatures
model.add_constraint('pt0.tBat', lower=0.0, upper = 45.0)
model.add_constraint('pt0.tProp', lower=-10.0, upper = 80.0)
model.add_constraint('pt1.tBat', lower=0.0, upper = 45.0)
model.add_constraint('pt1.tProp', lower=-10.0, upper = 80.0)
#model.add_constraint('tMain', lower=-40.0, upper = 85.0) """

#run the ExternalCode Component once and record initial values
prob.setup(check=True)

""" prob.run_model()

tBat_c1 =  prob['tBat_c']
tBat_h1 =  prob['tBat_h']
tProp_c1 =  prob['tProp_c']
tProp_h1 =  prob['tProp_h']
tMain_c1 =  prob['tMain_c']
tMain_h1 =  prob['tMain_h'] """

# find optimal solution with simple GA driver
prob.driver = om.SimpleGADriver()
prob.driver.options['bits'] = {'length':5, 'eps':6, 'r_bat': 3, 'R_m': 5, 'R_p': 5, 'R_s':5}
prob.driver.options['max_gen'] = 10
#prob.driver.options['run_parallel'] = 'true'
prob.driver.options['debug_print'] = ['desvars']
prob.driver.add_recorder(om.SqliteRecorder("mdp_genopt.sql"))

model.add_design_var('length', lower = 0.0, upper=0.254)
model.add_design_var('eps', lower = 0.02, upper=0.8)
model.add_design_var('r_bat', lower = 0.0, upper=1.0)
model.add_design_var('R_m', lower = 1., upper=250.0)
model.add_design_var('R_p', lower = 1., upper=250.0)
model.add_design_var('R_s', lower = 1., upper=250.0)

prob.model.add_objective('obj_p')

#Run optimization
tStart = time.time()
prob.setup(check=True)
prob.run_driver()

#Record final temperatures
tBat_c2 =  prob['tBat_c']
tBat_h2 =  prob['tBat_h']
tProp_c2 =  prob['tProp_c']
tProp_h2 =  prob['tProp_h']
tMain_c2 =  prob['tMain_c']
tMain_h2 =  prob['tMain_h']

""" print("Temperatures before optimization:, tBat_c1={}, tProp_c1={}, tMain_c1={}, tBat_h1={}, tProp_h1={}, tMain_h1={}".format(tBat_c1, tProp_c1, tMain_c1, tBat_h1, tProp_h1, tMain_h1)) 
print("Temperatures after optimization:,  tBat_c2={}, tProp_c2={}, tMain_c2={}, tBat_h2={}, tProp_h2={}, tMain_h2={}".format(tBat_c2, tProp_c2, tMain_c2, tBat_h2, tProp_h2, tMain_h2))
print("Final design variables: batH = {}, propH = {}, eps1={}, alp1={}, eps2={}, GlMain={}, GlProp={}, GlTether={}, ci1={}, ci2={}, ci3={}, ci4={}, ci5={}, ci6={}, ci7={}, ci8={}, ci9={}, ci10={}, ci11={}, ci12={}".format (prob['batH'], prob['propH'], prob['eps1'], prob['alp1'], prob['eps2'], prob['GlMain'], prob['GlProp'], prob['GlTether'],
prob['ci1'], prob['ci2'], prob['ci3'], prob['ci4'], prob['ci5'], prob['ci6'], prob['ci7'], prob['ci8'], prob['ci9'], prob['ci10'], prob['ci11'], prob['ci12']))
print("Objective value:", prob['obj_p']) """

print("Optimization run time in minutes:", (time.time()-tStart)/60)

# save result (objective and constraints) to a pickle

data = {'obj.val': prob['length'], 'eps': prob['eps'], 'R_m': prob['R_m'],
'R_p': prob['R_p'], 'R_s': prob['R_s'], 'r_bat': prob['r_bat'], 'tBat_c':tBat_c2, 'tBat_h': tBat_h2,
'tProp_c':tProp_c2, 'tProp_h':tProp_h2}

pprint(data)
pickle.dump(data, open('mdp_genopt.p', 'wb'))