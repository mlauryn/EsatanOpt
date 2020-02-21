#Python script for optimization of MAT remote unit thermal model @cold analysis case using surrogate model
import os, time
from openmdao.api import Problem, Group, IndepVarComp, ExternalCode, ScipyOptimizeDriver, SimpleGADriver, ExecComp, ExplicitComponent
import numpy as np
import openmdao.api as om 

data_1 = np.loadtxt('TrainingData_1.csv', delimiter=',')
data_2 = np.loadtxt('TrainingData_2.csv', delimiter=',')

#Cold case thermal model surrogate
metamod_1 = om.MetaModelUnStructuredComp()
metamod_1.add_input('eps', 0.)
metamod_1.add_input('alp', 0.)
metamod_1.add_input('batH', 0.)
metamod_1.add_input('propH', 0.)
metamod_1.add_output('tBat', 0., surrogate=om.KrigingSurrogate())
metamod_1.add_output('tProp', 0., surrogate=om.KrigingSurrogate())

# train the surrogate
metamod_1.options['train:eps'] = data_1[:,0]
metamod_1.options['train:alp'] = data_1[:,1]
metamod_1.options['train:batH'] = data_1[:,2]
metamod_1.options['train:propH'] = data_1[:,3]
metamod_1.options['train:tBat'] = data_1[:,4]
metamod_1.options['train:tProp'] = data_1[:,5]

#Hot case thermal model surrogate
metamod_2 = om.MetaModelUnStructuredComp()
metamod_2.add_input('eps', 0.)
metamod_2.add_input('alp', 0.)
metamod_2.add_output('tBat', 0., surrogate=om.KrigingSurrogate())
metamod_2.add_output('tProp', 0., surrogate=om.KrigingSurrogate())

# train the surrogate
metamod_2.options['train:eps'] = data_2[:,0]
metamod_2.options['train:alp'] = data_2[:,1]
metamod_2.options['train:tBat'] = data_2[:,2]
metamod_2.options['train:tProp'] = data_2[:,3]

prob = Problem()
model = prob.model

# create and connect inputs and outputs
indeps = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
indeps.add_output('batH', val=0.2)
indeps.add_output('propH', val=0.2)
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
model.add_subsystem('mm_1', metamod_1, promotes_inputs=['*'], promotes_outputs=[('tBat','tBat_c'), ('tProp','tProp_c')])
model.add_subsystem('mm_2', metamod_2, promotes_inputs=['*'], promotes_outputs=[('tBat','tBat_h'), ('tProp','tProp_h')])
model.add_subsystem('obj', ExecComp('P = batH + propH'), promotes=['*'])

#run the ExternalCode Component once and record initial values
""" prob.setup(check=True)
prob.run_model() """

prob.model.add_design_var('eps', lower = 0.02, upper=0.8)
prob.model.add_design_var('alp', lower = 0.23, upper=0.48)
prob.model.add_design_var('batH', lower = 0.0, upper=1.0)
prob.model.add_design_var('propH', lower = 0.0, upper=1.0)
""" prob.model.add_design_var('GlBat1', lower = 0.4, upper=26.0)
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

#objective function is to minimize heating power
prob.model.add_objective('P')

prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer']='SLSQP'
prob.driver.options['disp'] = True
#prob.driver.opt_settings = {'eps': 1.0e-6, 'ftol':1e-04,} 

#constraint for  temperatures
prob.model.add_constraint('tBat_c', lower=0.0, upper = 45.0)
prob.model.add_constraint('tProp_c', lower=-10.0, upper = 80.0)
#prob.model.add_constraint('tMain_c', lower=-40.0, upper = 85.0)
prob.model.add_constraint('tBat_h', lower=0.0, upper = 45.0)
prob.model.add_constraint('tProp_h', lower=-10.0, upper = 80.0)
#prob.model.add_constraint('tMain_h', lower=-40.0, upper = 85.0)
#prob.model.add_constraint('tTether_h', lower=-40.0, upper = 50.0)

#Run optimization
tStart = time.time()
prob.setup(check=True)
prob.run_driver()

#Record final temperatures
tBat_c2 =  prob['tBat_c']
tBat_h2 =  prob['tBat_h']
tProp_c2 =  prob['tProp_c']
tProp_h2 =  prob['tProp_h']

print("Temperatures after optimization:,  tBat_c2={}, tProp_c2={}, tBat_h2={}, tProp_h2={}".format(tBat_c2, tProp_c2, tBat_h2, tProp_h2))
print("Final design variables: batH = {}, propH = {}, eps={}, alp={}".format (prob['batH'], prob['propH'], prob['eps'], prob['alp']))

print("Optimization run time in minutes:", (time.time()-tStart)/60)