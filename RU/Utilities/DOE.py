import openmdao.api as om
import numpy as np
from Pre_process import nodes, idx_dict
from Thermal_MDF import Thermal_MDF

npts = 2 # num of point
num = 40 # num of samples
model_name = 'RU_v4_detail'
#keys = list(groups.keys()) # import all nodes?
keys = ['Box:outer', 'Panel_outer:solar_cells', 'Panel_inner:solar_cells', 'Panel_body:solar_cells'] # define faces to include in radiative analysis

model = Thermal_MDF(npts=npts, labels=keys, model=model_name)
prob = om.Problem(model=model)

model.add_design_var('Spacer5', lower=0.25, upper=237.)
model.add_design_var('Spacer1', lower=0.25, upper=237.)
model.add_design_var('Body_panel', lower=0.004, upper=.1)
model.add_design_var('Hinge_middle', lower=0.02, upper=.1)
model.add_design_var('Hinge_outer', lower=0.02, upper=.1)

#model.add_design_var('cr', lower=0.0, upper=1., indices=list(idx['Panel_body:solar_cells'])) # only body solar cells are selected here
#model.add_design_var('alp_r', lower=0.07, upper=0.94, indices=list(idx['Box:outer'])) # optimize absorbptivity for structure
model.add_design_var('Box:outer', lower=0.02, upper=0.94) # optimize emissivity of structure
model.add_design_var('QI', lower = 0.25, upper=7., indices=[-1, -2, -7, -8, -10])
model.add_design_var('phi', lower=0., upper=90.)

model.add_objective('obj')

prob.driver = om.DOEDriver(om.LatinHypercubeGenerator(samples=num))
name = model_name + '_doe_' + str(num) + '.sql'
prob.driver.add_recorder(om.SqliteRecorder(name))

prob.setup(check=True)

prob.run_driver()

prob.cleanup()

cr = om.CaseReader(name)
cases = cr.list_cases('driver')

print(len(cases))

""" values = []
for case in cases:
    outputs = cr.get_case(case).outputs
    values.append((outputs['Spacer5'], outputs['Spacer1'], outputs['Body_panel'],
    outputs['Hinge_middle'], outputs['Hinge_outer'], outputs['Box:outer'],
    outputs['QI'], outputs['phi'])) """

#print("\n".join(["x: %5.2f, y: %5.2f, f_xy: %6.2f" % xyf for xyf in values]))