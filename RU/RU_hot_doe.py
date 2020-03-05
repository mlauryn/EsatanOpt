#Python script for optimization of MAT remote unit thermal model @cold analysis case
import os, time
import openmdao.api as om
import numpy as np
import RU_hot_esatan as ru

prob = om.Problem()
model = prob.model

# create and connect inputs and outputs
indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
indeps.add_output('length', val=0.2)
indeps.add_output('eff', val=0.28)
indeps.add_output('P_un', val=0.2)
indeps.add_output('eps', val=0.2)
#indeps.add_output('alp', val=0.4)
indeps.add_output('GlMain', val=0.04)
indeps.add_output('GlProp', val=0.04)
indeps.add_output('GlTether', val=0.04)
indeps.add_output('GlPanel', val=0.04)


model.add_subsystem('esatan', ru.RU_hot(), promotes=['*'])

model.add_design_var('length', lower = 0.0, upper=0.254)
model.add_design_var('eff', lower = 0.25, upper=0.32)
model.add_design_var('eps', lower = 0.02, upper=0.8)
#model.add_design_var('alp', lower = 0.23, upper=0.48)
prob.model.add_design_var('P_un', lower = 0.0, upper=1.0)
prob.model.add_design_var('GlMain', lower = 0.004, upper=1.0)
prob.model.add_design_var('GlProp', lower = 0.004, upper=1.0)
prob.model.add_design_var('GlTether', lower = 0.004, upper=1.0)
prob.model.add_design_var('GlPanel', lower = 0.004, upper=1.0)

model.add_objective('tBat')
model.add_objective('tProp')
model.add_objective('tBPanel')
model.add_objective('tDPanel')

prob.driver = om.DOEDriver(om.LatinHypercubeGenerator(num_samples=20))
prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

prob.setup(check=True)
prob.run_driver()
prob.cleanup()

cr = om.CaseReader("cases.sql")
cases = cr.list_cases('driver')

values = []
for case in cases:
    outputs = cr.get_case(case).outputs
    values.append((outputs['eps'], outputs['length'], outputs['eff'] outputs['P_un'], 
    outputs['GlMain'], outputs['GlProp'], outputs['GlTether'], outputs['GlPanel'],
    outputs['tBat'], outputs['tProp'], outputs['tBPanel'], outputs['tDPanel']))

data = np.reshape(values, (len(cases), 13))
#print(data)
np.savetxt('RUh_TrainingData_n=20.csv', data, delimiter=',')