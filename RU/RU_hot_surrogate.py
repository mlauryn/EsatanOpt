#Python script for optimization of MAT remote unit thermal model @hot analysis case using surrogate model
import os, time
from openmdao.api import Problem, Group, IndepVarComp, ExternalCode, ScipyOptimizeDriver, SimpleGADriver, ExecComp, ExplicitComponent
import numpy as np
import openmdao.api as om 

data = np.loadtxt('TrainingData_2.csv', delimiter=',')
print(data)

metamod = om.MetaModelUnStructuredComp()
metamod.add_input('eps', 0.)
metamod.add_input('alp', 0.)
metamod.add_output('tBat', 0., surrogate=om.KrigingSurrogate())

# train the surrogate
metamod.options['train:eps'] = data[:,0]
metamod.options['train:alp'] = data[:,1]
metamod.options['train:tBat'] = data[:,2]

prob = Problem()
model = prob.model

# create and connect inputs and outputs
indeps = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
""" indeps.add_output('batH', val=0.2)
indeps.add_output('propH', val=0.2) """
indeps.add_output('eps', val=0.2)
indeps.add_output('alp', val=0.4)
""" indeps.add_output('GlBat1', val=0.4)
indeps.add_output('GlBat2', val=0.4)
indeps.add_output('GlMain', val=0.04)
indeps.add_output('GlProp', val=0.04)
indeps.add_output('GlTether', val=0.04)
indeps.add_output('ci1', val=0.4)
indeps.add_output('ci2', val=0.4)
indeps.add_output('ci3', val=0.4)
indeps.add_output('ci4', val=0.4)
indeps.add_output('ci5', val=0.4)
indeps.add_output('ci6', val=0.4)
indeps.add_output('ci7', val=0.4)
indeps.add_output('ci8', val=0.4)
indeps.add_output('ci9', val=0.4)
indeps.add_output('ci10', val=0.4)
indeps.add_output('ci11', val=0.4)
indeps.add_output('ci12', val=0.4)  """
model.add_subsystem('mm', metamod, promotes=['*'])
#model.add_subsystem('obj', ExecComp('T = 0 - tBat'), promotes=['*'])

#run the ExternalCode Component once and record initial values
""" prob.setup(check=True)
prob.run_model() """

prob.model.add_design_var('eps', lower = 0.02, upper=0.8)
prob.model.add_design_var('alp', lower = 0.23, upper=0.48)
""" prob.model.add_design_var('batH', lower = 0.0, upper=1.0)
prob.model.add_design_var('propH', lower = 0.0, upper=1.0)
prob.model.add_design_var('GlBat1', lower = 0.4, upper=26.0)
prob.model.add_design_var('GlBat2', lower = 0.4, upper=26.0)
prob.model.add_design_var('GlMain', lower = 0.004, upper=1.0)
prob.model.add_design_var('GlProp', lower = 0.004, upper=1.0)
prob.model.add_design_var('GlTether', lower = 0.004, upper=1.0)
prob.model.add_design_var('ci1', lower = 0.013, upper=0.072)
prob.model.add_design_var('ci2', lower = 0.015, upper=0.084)
prob.model.add_design_var('ci3', lower = 0.015, upper=0.084)
prob.model.add_design_var('ci4', lower = 0.008, upper=0.026)
prob.model.add_design_var('ci5', lower = 0.008, upper=0.026)
prob.model.add_design_var('ci6', lower = 0.013, upper=0.072)
prob.model.add_design_var('ci7', lower = 0.013, upper=0.072)
prob.model.add_design_var('ci8', lower = 0.013, upper=0.072)
prob.model.add_design_var('ci9', lower = 0.015, upper=0.084)
prob.model.add_design_var('ci10', lower = 0.008, upper=0.026)
prob.model.add_design_var('ci11', lower = 0.008, upper=0.026)
prob.model.add_design_var('ci12', lower = 0.015, upper=0.084) """

#objective function is to minimize battery temp
prob.model.add_objective('tBat')

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
print(prob['alp'])
print("Optimization run time in minutes:", (time.time()-tStart)/60)