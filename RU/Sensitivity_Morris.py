import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import morris
import RU_hot_esatan
import openmdao.api as om 


problem = {
    'num_vars': 18,
    'names': ['eps', 'alp', 'GlBat', 'GlMain', 'GlProp', 'GlTether', 
             'ci1', 'ci2', 'ci3', 'ci4', 'ci5','ci6','ci7','ci8','ci9','ci10','ci11','ci12',],
    'bounds': [[0.02, 0.8],
               [0.23, 0.9],
               [0.4, 26],
               [0.004, 1],
               [0.004, 1],
               [0.004, 1],
               [0.013, 0.072],
               [0.015, 0.084],
               [0.015, 0.084],
               [0.008, 0.026],
               [0.008, 0.026],
               [0.013, 0.072],
               [0.013, 0.072],
               [0.013, 0.072],
               [0.015, 0.084],
               [0.008, 0.026],
               [0.008, 0.026],
               [0.015, 0.084]]
}
X = morris.sample(problem, 1)
np.savetxt('morris_sample.csv', X, delimiter=',', header = 'eps, alp, GlBat, GlMain, GlProp, GlTether, ci1, ci2, ci3, ci4, ci5, ci6, ci7, ci8, ci9, ci10, ci11, ci12',
comments = '')

# this file contains design variable inputs in CSV format
""" with open('morris_sample.csv', 'r') as f:
    print(f.read()) """

prob = om.Problem()
model = prob.model

# create and connect inputs and outputs
indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
indeps.add_output('eps', val=0.2)
indeps.add_output('alp', val=0.4)
indeps.add_output('GlBat', val=0.4)
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
indeps.add_output('ci12', val=0.4) 

model.add_subsystem('esatan', RU_hot_esatan.RU_hot(), promotes=['*'])

model.add_design_var('eps', lower = 0.02, upper=0.8)
model.add_design_var('alp', lower = 0.23, upper=0.48)
prob.model.add_design_var('GlBat', lower = 0.4, upper=26.0)
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
prob.model.add_design_var('ci12', lower = 0.015, upper=0.084)
model.add_objective('tBat')
model.add_objective('tProp')

prob.setup(check=True)

prob.driver = om.DOEDriver(om.CSVGenerator('morris_sample.csv'))
prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))


prob.run_driver()
prob.cleanup()

cr = om.CaseReader("cases.sql")
cases = cr.list_cases('driver')

values = []
for case in cases:
    outputs = cr.get_case(case).outputs
    values.append((outputs['tBat'], outputs['tProp']))

data = np.reshape(values, (len(cases), 2))

Y = data[:, 0]

from SALib.analyze import morris
Si = morris.analyze(problem, X, Y, conf_level=0.95,
                     print_to_console=True, num_levels=4)