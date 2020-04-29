import openmdao.api as om
import numpy as np
from Pre_process import nodes, idx_dict
from Thermal_MDF import Thermal_MDF

npts = 2
model_name = 'RU_v4_detail'
#keys = list(groups.keys()) # import all nodes?
keys = ['Box:outer', 'Panel_outer:solar_cells', 'Panel_inner:solar_cells', 'Panel_body:solar_cells'] # define faces to include in radiative analysis

model_dir = './Esatan_models/' + model_name
data = model_dir+'/nodes_output.csv'
nn, groups = nodes(data=data)
nodes_list = sum([groups[group] for group in keys], [])
#print(nodes)

# index dictionary or radiative nodes_list
idx = idx_dict(sorted(nodes_list), groups)

# indices for solar cells
sc_idx = sum([idx[keys] for keys in ['Panel_outer:solar_cells', 'Panel_inner:solar_cells', 'Panel_body:solar_cells']], [])

model = Thermal_MDF(npts=npts, labels=keys, model=model_name)
prob = om.Problem(model=model)

model.add_design_var('Spacer5', lower=0.25, upper=237.)
model.add_design_var('Spacer1', lower=0.25, upper=237.)
model.add_design_var('Body_panel', lower=0.004, upper=.1)
model.add_design_var('Hinge_middle', lower=0.02, upper=.1)
model.add_design_var('Hinge_outer', lower=0.02, upper=.1)
#model.add_design_var('cr', lower=0.0, upper=1., indices=list(idx['Panel_body:solar_cells'])) # only body solar cells are selected here
model.add_design_var('alp_r', lower=0.07, upper=0.94, indices=list(idx['Box:outer'])) # optimize absorbptivity for structure
model.add_design_var('Box:outer', lower=0.02, upper=0.94) # optimize emissivity of structure
model.add_design_var('QI', lower = 0.25, upper=7., indices=[-1, -2, -7, -8, -10])
model.add_design_var('phi', lower=0., upper=90.)

model.add_constraint('bat_lwr.KS', upper=0.0)
model.add_constraint('bat_upr.KS', upper=0.0)
model.add_constraint('prop_upr.KS', upper=0.0)
model.add_constraint('prop_lwr.KS', upper=0.0)

model.add_objective('obj')

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer']='SLSQP'
prob.driver.options['disp'] = True
prob.driver.options['maxiter'] = 70
prob.driver.options['tol'] = 1.0e-4
prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']
prob.driver.add_recorder(om.SqliteRecorder("ru_v4_detail_mstart_1.sql"))

prob.setup(check=True)

""" cr = om.CaseReader('RU_v4_detail_doe_40.sql')
cases = cr.list_cases('driver')
num_cases = len(cases) """

""" # run optimizer for doe cases
for i in range(num_cases): 
    # Load the case
    case = cr.get_case(cases[i])
    prob.load_case(case) """

# set initial values for solar cells and radiators
prob['cr'][sc_idx] = 1.0
prob['alp_r'][list(idx['Box:outer'])] = 0.5
prob['QI'][[-1]] = 0.2
prob['QI'][[-4]] = 0.3

#print(cases[i])
prefix = 'Opt_run' + '0'

prob.run_driver(case_prefix=prefix)



#prob.model.list_inputs(print_arrays=True)
print(prob['T']-273.)