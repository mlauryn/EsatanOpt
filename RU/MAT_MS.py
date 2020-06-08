import os
import openmdao.api as om
import numpy as np
from MS import MainSP
from Pre_process import nodes, inits, idx_dict, parse_cond
from fnmatch import fnmatch
import pandas as pd
from time import time

# number of design points
npts = 1

# model version name
model_name = 'MAT_v2_8'

# define faces to include in radiative analysis
#geom = list(groups.keys()) # import all nodes?
geom = {'SP_Xplus' : 0.85,
    'SP_Xminus' : 0.85,
    'SP_Yplus' : 0.85,
    'SP_Yplus' : 0.85,
    'SP_Yminus' : 0.85,
    'SP_Zplus' : 0.85,
    'SP_Zminus' : 0.85,
    'SolarArray' : 0.89,
    'Propulsion' : 0.89,
    'Esail' : 0.022,
    'BottomPlate' : 0.04,
    'Telescope' : 0.88,
    'StarTracker': 0.88,
    'Instrument': 0.88,
    #'Instrument_inner'
    }

fpath = os.path.dirname(os.path.realpath(__file__))
model_dir = fpath + '/Esatan_models/' + model_name
data = model_dir+'/nodes_output.csv'

nn, groups, output, area = nodes(data=data)
QI_init, QS_init = inits(data=data)
rad_nodes = sum([groups[group] for group in geom], [])

# import user-defined conductors
MS_cond = os.path.join(model_dir, 'cond_report.txt')
MS_user_cond = parse_cond(MS_cond)

# index dictionary of radiative nodes_list
idx = idx_dict(sorted(rad_nodes), groups)

# user defined node groups
#groups.update({'radiator':[158]})# extra node to disipate heat in structure

# global indices for components with controlled heat disipation
equip = sum([groups[syst] for syst in [
    'Propulsion',
    'PCB',
    'Battery',
    'Instrument',
    'Esail',
    'TRx',
    'AOCS',
    'SolarArray'
    #'RW_Z',
    # radiators
    #'Reflectarray'
    #'SP_Xplus',
    #'SP_Xminus',
    #'SP_Yplus',
    #'SP_Yminus',
    #'SP_Zplus',
    #'SP_Zminus', 
    ]], [])

# global indices into flattened array
flat_indices = np.arange(0,(nn+1)*npts).reshape((nn+1,npts))

# remote unit model group instance
model = MainSP(npts=npts, labels=geom, model=model_name)

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

model.add_design_var('SP_Xplus', lower=0.02, upper=0.94)
model.add_design_var('SP_Xminus', lower=0.02, upper=0.94)
model.add_design_var('SP_Yplus', lower=0.02, upper=0.94)
model.add_design_var('SP_Yminus', lower=0.02, upper=0.94)
model.add_design_var('SP_Zplus', lower=0.02, upper=0.94)
model.add_design_var('SP_Zminus', lower=0.02, upper=0.94)
model.add_design_var('Esail', lower=0.02, upper=0.94)
model.add_design_var('Propulsion', lower=0.02, upper=0.94)
model.add_design_var('QI', lower = 0., upper=10., indices=(flat_indices[equip,:]).ravel())
model.add_design_var('phi', lower=0., upper=45.)

radiator_surf = sum([idx[surf] for surf in [
    'SP_Xplus',
    'SP_Xminus',
    'SP_Yplus',
    'SP_Yminus',
    'SP_Zplus',
    'SP_Zminus']], [])

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
prob.driver.add_recorder(om.SqliteRecorder('./Cases/' + model_name + '.sql'))

prob.setup(check=True)

# initial values for some input variables
prob['phi'] = [0.]
prob['dist'] = [1.]

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
prob.run_model()
#prob.run_driver()
run_time = time() - run_start
print('Run Time:', run_time, 's')

output['T_res'] = prob['T'][1:,0]-273.15
#output['T_res2'] = prob['T'][1:,1]-273.15
output['abs'] = output['T_ref']-output['T_res']
#output['rel'] = output['abs']/output['T_ref']
print(output)
output.to_csv('./Cases/' + model_name + '_out.csv')

""" temp_MS = pd.DataFrame(data=prob['T'][1:,:]-273.15, index=output.index)
temp_MS['label'] = output[0]
temp_MS.to_csv('./Cases/MAT_MS_out.csv') """

#print(best_case)

#totals = prob.compute_totals()#of=['T'], wrt=['Spacer5'])
#print(totals)
#check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=True, step=1e-04)
#prob.check_totals(of=['equip_upr.KS'], wrt='phi', compact_print=True)

#prob.model.list_outputs(print_arrays=False)
