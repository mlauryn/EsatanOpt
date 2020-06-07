import os
import openmdao.api as om
import numpy as np
from MS import MainSP
from Pre_process import nodes, inits, idx_dict, parse_cond
from fnmatch import fnmatch
import pandas as pd
from time import time

# number of design points
npts = 2

# model version name
model_name = 'MAT'

# define faces to include in radiative analysis
#face_IDs = list(groups.keys()) # import all nodes?
face_IDs = [
    'SP_Xplus_upr',
    'SP_Xminus_upr',
    'SP_Yplus_upr',
    'SP_Yplus_lwr',
    'SP_Yminus_upr',
    'SP_Zplus_upr',
    'SP_Zminus_upr',
    'SolarArray',
    'Propulsion_top',
    'Esail_top',
    'Esail_bot',
    'BottomPlate_upr',
    'Telescope_outer',
    'StarTracker_outer',
    'Instrument_outer',
    #'Instrument_inner'
    ]

fpath = os.path.dirname(os.path.realpath(__file__))
model_dir = fpath + '/Esatan_models/' + model_name
data = model_dir+'/nodes_output.csv'

nn, groups, output = nodes(data=data)
QI_init, QS_init = inits(data=data)
rad_nodes = sum([groups[group] for group in face_IDs], [])

# import user-defined conductors
MS_cond = os.path.join(model_dir, 'cond_report.txt')
MS_user_cond = parse_cond(MS_cond)

# index dictionary of radiative nodes_list
idx = idx_dict(sorted(rad_nodes), groups)

# user defined node groups
#groups.update({'radiator':[158]})# extra node to disipate heat in structure

# global indices for components with controlled heat disipation
equip = sum([groups[syst] for syst in [
    'Propulsion_bot',
    'PCB',
    'Battery',
    'Instrument_inner',
    #'Esail_bot',
    'TRx',
    'AOCS',
    #'RW_Z',
    # radiators
    'Reflectarray'
    #'SP_Xplus_upr',
    #'SP_Xminus_upr',
    #'SP_Yplus_upr',
    #'SP_Yminus_upr',
    #'SP_Zplus_upr',
    #'SP_Zminus_upr', 
    ]], [])

# global indices into flattened array
flat_indices = np.arange(0,(nn+1)*npts).reshape((nn+1,npts))

# remote unit model group instance
model = MainSP(npts=npts, labels=face_IDs, model=model_name)

####################################
# otpimization problem formulation #
####################################

# design variables
# seperate hinges because they have different bounds
MS_conn = [cond for cond in MS_user_cond if fnmatch(cond['cond_name'], 'Hinge*')==False]
MS_hinges = [cond for cond in MS_user_cond if fnmatch(cond['cond_name'], 'Hinge*')]

for cond in MS_conn:
    model.add_design_var(cond['cond_name'], lower=0.001, upper=10. ) # all conductors except hinges
for cond in MS_hinges:
    model.add_design_var(cond['cond_name'], lower=0.01, upper=.1 ) # solar array hinges

model.add_design_var('SP_Xplus_upr', lower=0.02, upper=0.94)
model.add_design_var('SP_Xminus_upr', lower=0.02, upper=0.94)
model.add_design_var('SP_Yplus_upr', lower=0.02, upper=0.94)
model.add_design_var('SP_Yminus_upr', lower=0.02, upper=0.94)
model.add_design_var('SP_Zplus_upr', lower=0.02, upper=0.94)
model.add_design_var('SP_Zminus_upr', lower=0.02, upper=0.94)
model.add_design_var('Esail_bot', lower=0.02, upper=0.94)
model.add_design_var('Propulsion_top', lower=0.02, upper=0.94)
model.add_design_var('QI', lower = 0., upper=10., indices=(flat_indices[equip,:]).ravel())
model.add_design_var('phi', lower=0., upper=45.)

radiator_surf = sum([idx[surf] for surf in [
    #'SP_Xplus_upr',
    'SP_Xminus_upr',
    'SP_Yplus_upr',
    'SP_Yminus_upr',
    'SP_Zplus_upr',
    'SP_Zminus_upr']], [])

model.add_design_var('alp_r', lower=0.07, upper=0.94, indices=radiator_surf)


# MS constraints
model.add_constraint('bat_lwr.KS', upper=0.0)
model.add_constraint('bat_upr.KS', upper=0.0)
model.add_constraint('prop_upr.KS', upper=0.0)
model.add_constraint('prop_lwr.KS', upper=0.0)
model.add_constraint('equip_upr.KS', upper=0.0)
model.add_constraint('equip_lwr.KS', upper=0.0)

model.add_constraint('obc_pwr.KS', upper=0.0)
model.add_constraint('prop_pwr.KS', upper=0.0)
#model.add_constraint('ins_pwr.KS', upper=0.0)
model.add_constraint('aocs_pwr.KS', upper=0.0)
#model.add_constraint('es_pwr.KS', upper=0.0)

# objective function
model.add_objective('P_trx')

prob = om.Problem(model=model)
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer']='SLSQP'
prob.driver.options['disp'] = True
prob.driver.options['maxiter'] = 300
prob.driver.options['tol'] = 1.0e-4
#prob.driver.opt_settings['minimizer_kwargs'] = {"method": "SLSQP", "jac": True}
#prob.driver.opt_settings['stepsize'] = 0.01
prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']
prob.driver.add_recorder(om.SqliteRecorder('./Cases/MAT_MS' +'.sql'))

prob.setup(check=True)

# initial values for some input variables
prob['phi'] = [15.,15.]
prob['dist'] = [2.75, 1.]

# load case?
""" cr = om.CaseReader('./Cases/RU_v4_detail_mstart_30.sql')
cases = cr.list_cases('driver')
num_cases = len(cases)
print(num_cases) """

# Load the last case written?
""" last_case = cr.get_case(cases[num_cases-1])
best_case = cr.get_case('Opt_run3_rank0:ScipyOptimize_SLSQP|79')
prob.load_case(best_case) """

run_start = time()
#prob.run_model()
prob.run_driver()
run_time = time() - run_start
print('Run Time:', run_time, 's')

#output['T_res'] = prob['T'][1:,0]-273.15
#output['T_res2'] = prob['T'][1:,1]-273.15
#output['abs'] = output['T_ref']-output['T_res']
#output['rel'] = output['abs']/output['T_ref']
temp_MS = pd.DataFrame(data=prob['T'][1:,:]-273.15, index=output.index)
temp_MS['label'] = output[0]
temp_MS.to_csv('./Cases/MAT_MS_out.csv')
#print(output)
#output.to_csv('./Cases/' + model_name + '_out.csv')
#print(best_case)

#totals = prob.compute_totals()#of=['T'], wrt=['Spacer5'])
#print(totals)
#check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=True, step=1e-04)
#prob.check_totals(of=['equip_upr.KS'], wrt='phi', compact_print=True)

#prob.model.list_outputs(print_arrays=False)
