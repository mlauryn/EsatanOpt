#Python script for optimization of MAT remote unit thermal model @cold analysis case using surrogate model
import os, time
from openmdao.api import Problem, Group, IndepVarComp, ExternalCode, ScipyOptimizeDriver, SimpleGADriver, ExecComp, ExplicitComponent
import numpy as np
import openmdao.api as om 

train = np.loadtxt('RUc_TrainingData[m]_n=40.csv', delimiter=',')

# train the surrogate
ru_mm = om.MetaModelUnStructuredComp(default_surrogate=om.KrigingSurrogate())
ru_mm.add_input('eps', training_data=train[:,0])
ru_mm.add_input('length', training_data=train[:,1])
ru_mm.add_input('eff', training_data=train[:,2])
ru_mm.add_input('P_ht', training_data=train[:,3])
ru_mm.add_input('r_bat', training_data=train[:,4])
ru_mm.add_input('GlMain', training_data=train[:,5])
ru_mm.add_input('GlProp', training_data=train[:,6])
ru_mm.add_input('GlTether', training_data=train[:,7])
#ru_mm.add_input('GlPanel', val=np.zeros(200), training_data=train[:,8])

ru_mm.add_output('tBat', training_data=train[:,9])

prob = Problem()
model = prob.model

# create and connect inputs and outputs
indeps = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
indeps.add_output('eps', val=0.02)
indeps.add_output('length', val=0.2)
indeps.add_output('eff', val=0.25)
indeps.add_output('P_ht', val=0.2)
indeps.add_output('r_bat', val=0.5)
indeps.add_output('GlMain', val=0.004)
indeps.add_output('GlProp', val=0.004)
indeps.add_output('GlTether', val=0.004) 

model.add_subsystem('mm', ru_mm, promotes=['*'])
model.add_subsystem('obj', ExecComp('T = 0 - tBat'), promotes=['*'])

#run the ExternalCode Component once and record initial values
""" prob.setup(check=True)
prob.run_model() """

model.add_design_var('length', lower = 0.0, upper=0.254)
model.add_design_var('eff', lower = 0.25, upper=0.32)
model.add_design_var('eps', lower = 0.02, upper=0.8)
model.add_design_var('P_ht', lower = 0.0, upper=1.0)
model.add_design_var('r_bat', lower = 0.0, upper=1.0)
model.add_design_var('GlMain', lower = 0.004, upper=1.0)
model.add_design_var('GlProp', lower = 0.004, upper=1.0)
model.add_design_var('GlTether', lower = 0.004, upper=1.0)
#model.add_design_var('GlPanel', lower = 0.004, upper=1.0)

#objective function is to minimize battery temp
model.add_objective('T')

prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer']='SLSQP'
prob.driver.options['disp'] = True
#prob.driver.opt_settings = {'eps': 1.0e-6, 'ftol':1e-04,} 

#constraint for  temperatures
""" prob.model.add_constraint('tBat_c', lower=0.0, upper = 45.0)
prob.model.add_constraint('tProp_c', lower=-10.0, upper = 80.0)
prob.model.add_constraint('tMain_c', lower=-40.0, upper = 85.0)
prob.model.add_constraint('tBat_h', lower=0.0, upper = 45.0)
prob.model.add_constraint('tProp_h', lower=-10.0, upper = 80.0)
prob.model.add_constraint('tMain_h', lower=-40.0, upper = 85.0)
prob.model.add_constraint('tTether_h', lower=-40.0, upper = 50.0) """

#Run optimization
tStart = time.time()
prob.setup(check=True)
prob.run_driver()

#Record final temperatures
tBat_2 = prob['tBat']
print(tBat_2)
print(prob['eps'])
print("Optimization run time in minutes:", (time.time()-tStart)/60)