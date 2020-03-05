import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import morris
import RU_hot_esatan
import openmdao.api as om 


problem = {
    'num_vars': 9,
    'names': ['length', 'eff', 'P_un', 'eps', 'alp', 'GlMain', 'GlProp', 'GlTether', 
             'GlPanel'],
    'bounds': [[0.0, 0.254],
               [0.25, 0.32],
               [0.0, 1.0],
               [0.02, 0.8],
               [0.23, 0.48],
               [0.004, 1],
               [0.004, 1],
               [0.004, 1],
               [0.004, 1]]
}
X = morris.sample(problem, 1)
np.savetxt('morris_sample.csv', X, delimiter=',', header = 'length, eff, P_un, eps, alp, GlMain, GlProp, GlTether, GlPanel',
comments = '')

# this file contains design variable inputs in CSV format
""" with open('morris_sample.csv', 'r') as f:
    print(f.read()) """

prob = om.Problem()
model = prob.model

# create and connect inputs and outputs
indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
indeps.add_output('length', val=0.2)
indeps.add_output('eff', val=0.2)
indeps.add_output('P_un', val=0.2)
indeps.add_output('eps', val=0.2)
indeps.add_output('alp', val=0.4)
indeps.add_output('GlMain', val=0.04)
indeps.add_output('GlProp', val=0.04)
indeps.add_output('GlTether', val=0.04)
indeps.add_output('GlPanel', val=0.04)


model.add_subsystem('esatan', RU_hot_esatan.RU_hot(), promotes=['*'])

model.add_design_var('length', lower = 0.0, upper=0.254)
model.add_design_var('eff', lower = 0.25, upper=0.32)
model.add_design_var('P_un', lower = 0.0, upper=1.0)
model.add_design_var('eps', lower = 0.02, upper=0.8)
model.add_design_var('alp', lower = 0.23, upper=0.48)
model.add_design_var('GlMain', lower = 0.004, upper=1.0)
model.add_design_var('GlProp', lower = 0.004, upper=1.0)
model.add_design_var('GlTether', lower = 0.004, upper=1.0)
model.add_design_var('GlPanel', lower = 0.004, upper=1.0)
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