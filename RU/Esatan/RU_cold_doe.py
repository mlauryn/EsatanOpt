#Python script for creation and validation of a surrogate model for MAT remote unit thermal model @cold analysis case
import os, time
import openmdao.api as om
import numpy as np
import RU_cold_esatan as ru

prob = om.Problem()
model = prob.model

# create and connect inputs and outputs
indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
indeps.add_output('length', val=0.2)
indeps.add_output('eff', val=0.28)
indeps.add_output('P_ht', val=0.2)
indeps.add_output('r_bat', val=0.2)
indeps.add_output('eps', val=0.2)
#indeps.add_output('alp', val=0.4)
indeps.add_output('R_m', val=0.04)
indeps.add_output('R_p', val=0.04)
indeps.add_output('R_s', val=0.04)


model.add_subsystem('esatan', ru.RU_cold(), promotes=['*'])

model.add_design_var('length', lower = 0.0, upper=0.254)
model.add_design_var('eff', lower = 0.25, upper=0.32)
model.add_design_var('eps', lower = 0.02, upper=0.8)
#model.add_design_var('alp', lower = 0.23, upper=0.48)
model.add_design_var('P_ht', lower = 0.0, upper=1.0)
model.add_design_var('r_bat', lower = 0.0, upper=1.0)
model.add_design_var('R_m', lower = 0.004, upper=1.0)
model.add_design_var('R_p', lower = 0.004, upper=1.0)
model.add_design_var('R_s', lower = 0.004, upper=1.0)

model.add_objective('tBat')
model.add_objective('tProp')
model.add_objective('tBPanel')
model.add_objective('tDPanel')

prob.driver = om.DOEDriver(om.CSVGenerator('../Samples/RUc_LHsample[ese]_n=300.csv'))
prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

prob.setup(check=True)
prob.run_driver()
prob.cleanup()

cr = om.CaseReader("cases.sql")
cases = cr.list_cases('driver')

values = []
for case in cases:
    outputs = cr.get_case(case).outputs
    values.append((outputs['eps'], outputs['length'], outputs['eff'], outputs['P_ht'], 
    outputs['r_bat'], outputs['R_m'], outputs['R_p'], outputs['R_s'],
    outputs['tBat'], outputs['tProp'], outputs['tBPanel'], outputs['tDPanel']))

data = np.reshape(values, (len(cases), 12))
#print(data)
np.savetxt('../TrainingData/RUc_TrainingData[ese]_n=300.csv', data, delimiter=',')
