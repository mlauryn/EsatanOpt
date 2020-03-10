#Python script for MAT remote unit thermal model design of experiments

import openmdao.api as om
import numpy as np
from RU_esatan import RU_esatan
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS

case = 'hot' # hot or cold
num = 200 # num of samples

if case == 'hot':
    xlimits = np.array([[0.0, 0.254], [0.20, 0.40], [0.02, 0.8], [1, 250], [1, 250], [1, 250]])
elif case == 'cold':
    xlimits = np.array([[0.0, 0.254], [0.20, 0.40], [0.02, 0.8], [0.0, 1.0], [1, 250], [1, 250], [1, 250]])

sampling = LHS(xlimits=xlimits, criterion='ese')



x = sampling(num)

""" #also add variable limit values
edges = np.transpose(xlimits)
x = np.append(x, edges , axis=0) """

if case == 'hot':
    input_file = './Samples/RUh_LHsample[ese]_n={0}.csv'.format(num)
    np.savetxt(input_file, x, delimiter=',', header = 'length, eff, eps, R_m, R_p, R_s', comments = '')
elif case == 'cold':
    input_file = './Samples/RUc_LHsample[ese]_n={0}.csv'.format(num)
    np.savetxt(input_file, x, delimiter=',', header = 'length, eff, eps, r_bat, R_m, R_p, R_s', comments = '')

""" plt.plot(x[:, 2], x[:, 6], "o")
plt.xlabel("x")
plt.ylabel("y")
plt.show() """

prob = om.Problem()
model = prob.model

# create and connect inputs and outputs
indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
indeps.add_output('length', val=0.2)
indeps.add_output('eff', val=0.28)
indeps.add_output('eps', val=0.2)
indeps.add_output('R_m', val=0.04)
indeps.add_output('R_p', val=0.04)
indeps.add_output('R_s', val=0.04)

if case == 'cold':
    indeps.add_output('r_bat', val=0.0)
    indeps.add_output('ht_gain', val=1.0)
    indeps.add_output('q_s', val=150.)
else:
    indeps.add_output('ht_gain', val=0.)
    indeps.add_output('q_s', val=1365.)

model.add_subsystem('ru_tm', RU_esatan(), promotes=['*'])

model.add_design_var('length', lower = 0.0, upper=0.254)
model.add_design_var('eff', lower = 0.25, upper=0.325)
model.add_design_var('eps', lower = 0.02, upper=0.8)
model.add_design_var('R_m', lower = 1., upper=250.0)
model.add_design_var('R_p', lower = 1., upper=250.0)
model.add_design_var('R_s', lower = 1., upper=250.0)

if case == 'cold':
    model.add_design_var('r_bat', lower = 0.0, upper=1.0)

model.add_objective('tBat')
model.add_objective('tProp')
model.add_objective('tBPanel')
model.add_objective('tDPanel')

prob.driver = om.DOEDriver(om.CSVGenerator(input_file))
prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

prob.setup(check=True)
prob.run_driver()
prob.cleanup()

cr = om.CaseReader("cases.sql")
cases = cr.list_cases('driver')

values = []

if case == 'cold':
    for nt in cases:
        outputs = cr.get_case(nt).outputs
        values.append((outputs['eps'], outputs['length'], outputs['eff'], outputs['r_bat'],
        outputs['R_m'], outputs['R_p'], outputs['R_s'],
        outputs['tBat'], outputs['tProp'], outputs['tBPanel'], outputs['tDPanel']))
    
    data = np.reshape(values, (len(cases), len(xlimits)+4))
    output_file = './TrainingData/RUc_TrainingData[ese]_n={0}.csv'.format(num)
    np.savetxt(output_file, data, delimiter=',')
else:
    for nt in cases:
        outputs = cr.get_case(nt).outputs
        values.append((outputs['eps'], outputs['length'], outputs['eff'],
        outputs['R_m'], outputs['R_p'], outputs['R_s'],
        outputs['tBat'], outputs['tProp'], outputs['tBPanel'], outputs['tDPanel']))
    
    data = np.reshape(values, (len(cases), len(xlimits)+4))
    output_file = './TrainingData/RUh_TrainingData[ese]_n={0}.csv'.format(num)
    np.savetxt(output_file, data, delimiter=',')


