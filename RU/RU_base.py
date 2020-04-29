import os
import openmdao.api as om
import numpy as np
from Thermal_MDF import Thermal_MDF
from Pre_process import nodes, idx_dict

npts = 2
model_name = 'RU_v4_base'
#keys = list(groups.keys()) # import all nodes?
keys = ['Box',
    'Panel_outer:solar_cells',
    'Panel_inner:solar_cells',
    'Panel_body',]
    #'Panel_inner: back',
    #'Panel_outer:back'] # define faces to include in radiative analysis

fpath = os.path.dirname(os.path.realpath(__file__))
model_dir = fpath + '/Esatan_models/' + model_name
data = model_dir+'/nodes_output.csv'
nn, groups = nodes(data=data)
nodes_list = sum([groups[group] for group in keys], [])
#print(nodes)

# index dictionary or radiative nodes_list
idx = idx_dict(sorted(nodes_list), groups)

model = Thermal_MDF(npts=npts, labels=keys, model=model_name)
prob = om.Problem(model=model)

model.add_design_var('Spacer5', lower=0.25, upper=237.)
model.add_design_var('Spacer1', lower=0.25, upper=237.)
model.add_design_var('Hinge_inner', lower=0.004, upper=.1)
model.add_design_var('Hinge_middle', lower=0.02, upper=.1)
model.add_design_var('Hinge_outer', lower=0.02, upper=.1)
#model.add_design_var('cr', lower=0.0, upper=1., indices=list(idx['Panel_body:solar_cells'])) # only body solar cells are selected here
model.add_design_var('alp_r', lower=0.07, upper=0.94, indices=list(idx['Box'])) # optimize absorbptivity for structure
model.add_design_var('Box', lower=0.02, upper=0.94) # optimize emissivity of structure
#model.add_design_var('Panel_outer:back', lower=0.02, upper=0.94) # optimize emissivity of solar array back surface
#model.add_design_var('Panel_inner: back', lower=0.02, upper=0.94) # optimize emissivity of solar array back surface
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
#prob.driver.opt_settings['minimizer_kwargs'] = {"method": "SLSQP", "jac": True}
#prob.driver.opt_settings['stepsize'] = 0.01
prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']
prob.driver.add_recorder(om.SqliteRecorder(model_name+'.sql'))
#prob.add_recorder(om.SqliteRecorder(model_name+'_temps.sql'))
prob.driver.recording_options['includes'] = ['T']

prob.setup(check=True)

# indices for solar cells
sc_idx = sum([idx[keys] for keys in ['Panel_outer:solar_cells', 'Panel_inner:solar_cells', 'Panel_body']], [])

# initial values for solar cells and radiators

prob['cr'][sc_idx] = 1.0
prob['alp_r'][list(idx['Box'])] = 0.5
prob['QI'][[-1]] = 0.2
prob['QI'][[-4]] = 0.3

prob.run_model()
#prob.record_iteration('initial')
prob.run_driver()
#prob.record_iteration('final')
#print(prob['T']-273.)
#print(best_case)

#totals = prob.compute_totals()#of=['T'], wrt=['Spacer5'])
#print(totals)
#check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=True, step=1e-04)
#prob.check_totals(compact_print=True)

#prob.model.list_inputs(print_arrays=True)