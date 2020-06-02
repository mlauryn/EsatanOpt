import os
import openmdao.api as om
import numpy as np
from Thermal_MDF_unc import Thermal_MDF_unc
from Thermal_MDF import Thermal_MDF
from Pre_process import nodes, inits, idx_dict

# number of design points
npts = 1
model_name = 'RU_v5_1'

# define faces to include in radiative analysis
#face_IDs = list(groups.face_IDs()) # import all nodes?
face_IDs = [
    #'outer_surf'
    'Box:outer',
    'Panel_outer:solar_cells',
    'Panel_inner:solar_cells',
    #'Panel_body:solar_cells',
    'Panel_inner: back',
    'Panel_outer:back',
]

fpath = os.path.dirname(os.path.realpath(__file__))
model_dir = fpath + '/Esatan_models/' + model_name
data = model_dir+'/nodes_output.csv'
nn, groups, output = nodes(data=data)
QI_init, QS_init = inits(data=data)
nodes_list = sum([groups[group] for group in face_IDs], [])

# index dictionary or radiative nodes_list
idx = idx_dict(sorted(nodes_list), groups)

model = Thermal_MDF_unc(npts=npts, labels=face_IDs, model=model_name)
prob = om.Problem(model=model)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer']='SLSQP'
prob.driver.options['disp'] = True
prob.driver.options['maxiter'] = 70
prob.driver.options['tol'] = 1.0e-4
#prob.driver.opt_settings['minimizer_kwargs'] = {"method": "SLSQP", "jac": True}
#prob.driver.opt_settings['stepsize'] = 0.01
prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']
prob.driver.add_recorder(om.SqliteRecorder('./Cases/'+ model_name +'.sql'))

prob.setup(check=True)

# global indices for solar cell nodes
sc_idx = sum([idx[face_IDs] for face_IDs in [
    'Panel_outer:solar_cells',
    'Panel_inner:solar_cells',
    #'Panel_body:solar_cells'
    ]], [])

# initial values for some input variables
prob['cr'][sc_idx] = 1.0
prob['alp_r'][list(idx['Box:outer'])] = 0.5
prob['QI'] = QI_init
#prob['phi'] = 0.
#prob['QI'][[-4]] = 0.3
prob['dist'][0] = 3.

# load case?
""" cr = om.CaseReader('./Cases/RU_v4_detail_mstart_30.sql')
cases = cr.list_cases('driver')
num_cases = len(cases)
print(num_cases) """

# Load the last case written?
""" last_case = cr.get_case(cases[num_cases-1])
best_case = cr.get_case('Opt_run3_rank0:ScipyOptimize_SLSQP|79')
prob.load_case(best_case) """

prob.run_model()
#prob.run_driver()

output['T_res'] = prob['T'][1:,0]-273.15
#output['T_res2'] = prob['T'][1:,1]-273.15
output['abs'] = output['T_ref']-output['T_res']
output['rel'] = output['abs']/output['T_ref']
print(output)
#print(best_case)

#totals = prob.compute_totals()#of=['T'], wrt=['Spacer5'])
#print(totals)
#check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=True, step=1e-04)
#prob.check_totals(compact_print=True)

#prob.model.list_inputs(print_arrays=True)
