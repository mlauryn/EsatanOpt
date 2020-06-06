import os
import openmdao.api as om
import numpy as np
from MS import MainSP
from Pre_process import nodes, inits, idx_dict

# number of design points
npts = 1

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
    'Telescope_outer',
    'StarTracker_outer',
    'Instrument_outer',
    #'Instrument_inner'
    'BottomPlate_upr',
]

fpath = os.path.dirname(os.path.realpath(__file__))
model_dir = fpath + '/Esatan_models/' + model_name
data = model_dir+'/nodes_output.csv'

nn, groups, output = nodes(data=data)
QI_init, QS_init = inits(data=data)
rad_nodes = sum([groups[group] for group in face_IDs], [])

# index dictionary of radiative nodes_list
idx = idx_dict(sorted(rad_nodes), groups)

# global indices for solar cell nodes
solar_cells = sum([idx[array] for array in [
    'SolarArray'
    ]], [])

structure = sum([idx[geom] for geom in [
    'SP_Xplus_upr',
    'SP_Xminus_upr',
    'SP_Yplus_upr',
    'SP_Yminus_upr',
    'SP_Zplus_upr',
    'SP_Zminus_upr',
    ]], [])

optics = sum([idx[geom] for geom in [
    'Telescope_outer',
    'StarTracker_outer',
    ]], [])

inside = sum([idx[geom] for geom in [
    'SP_Yplus_lwr',
    'Instrument_outer',
    'BottomPlate_upr',
    ]], [])

# user defined node groups
#groups.update({'radiator':[158]})# extra node to disipate heat in structure

# global indices for components with controlled heat disipation
equip = sum([groups[syst] for syst in [
    'Propulsion_bot',
    'PCB',
    'Battery',
    'Instrument_inner',
    'Esail_bot',
    'TRx',
    'AOCS',
    'RW_Z',
    'SP_Yminus_lwr', # radiator
    ]], [])

# global indices into flattened array
flat_indices = np.arange(0,(nn+1)*npts).reshape((nn+1,npts))

# remote unit model group instance
model = MainSP(npts=npts, labels=face_IDs, model=model_name)

####################################
# otpimization problem formulation #
####################################

# design variables

#model.add_design_var('Body_panel', lower=0.004, upper=.1)
#model.add_design_var('Hinge_inner_1', lower=0.02, upper=.1)
#model.add_design_var('Hinge_inner_2', lower=0.02, upper=.1)
#model.add_design_var('Hinge_outer_1', lower=0.02, upper=.1)
#model.add_design_var('Hinge_outer_2', lower=0.02, upper=.1)
#model.add_design_var('cr', lower=0.0, upper=1., indices=list(idx['Panel_body:solar_cells'])) # only body solar cells are selected here
#model.add_design_var('alp_r', lower=0.07, upper=0.94, indices=list(idx['Box:outer'])) # optimize absorbptivity for structure
#model.add_design_var('Box:outer', lower=0.02, upper=0.94) # optimize emissivity of structure
#model.add_design_var('Panel_outer:back', lower=0.02, upper=0.94) # optimize emissivity of solar array back surface
#model.add_design_var('Panel_inner: back', lower=0.02, upper=0.94) # optimize emissivity of solar array back surface
model.add_design_var('QI', lower = 0., upper=7., indices=(flat_indices[equip,:]).ravel())
#model.add_design_var('phi', lower=0., upper=33.)

# constraints
model.add_constraint('bat_lwr.KS', upper=0.0)
model.add_constraint('bat_upr.KS', upper=0.0)
model.add_constraint('prop_upr.KS', upper=0.0)
model.add_constraint('prop_lwr.KS', upper=0.0)
model.add_constraint('obc_upr.KS', upper=0.0)
model.add_constraint('obc_lwr.KS', upper=0.0)
model.add_constraint('ins_upr.KS', upper=0.0)
model.add_constraint('ins_lwr.KS', upper=0.0)
model.add_constraint('es_upr.KS', upper=0.0)
model.add_constraint('es_lwr.KS', upper=0.0)
model.add_constraint('trx_upr.KS', upper=0.0)
model.add_constraint('trx_lwr.KS', upper=0.0)
model.add_constraint('aocs_upr.KS', upper=0.0)
model.add_constraint('aocs_lwr.KS', upper=0.0)

model.add_constraint('obc_pwr.KS', upper=0.0)
model.add_constraint('prop_pwr.KS', upper=0.0)
model.add_constraint('ins_pwr.KS', upper=0.0)
model.add_constraint('aocs_pwr.KS', upper=0.0)

# objective function
model.add_objective('P_trx')

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

# initial values for some input variables
prob['cr'][solar_cells] = 1.0
alp = prob['alp_r']
alp[structure] = 0.1
alp[inside] = 0.26
alp[optics] = 0.88
alp[idx['Propulsion_top']] = 0.72
alp[idx['Esail_top']] = 1.

prob['QI'] = QI_init
#prob['QS'] = QS_init
prob['phi'] = [30.]
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

#prob.run_model()
prob.run_driver()

output['T_res'] = prob['T'][1:,0]-273.15
#output['T_res2'] = prob['T'][1:,1]-273.15
#output['abs'] = output['T_ref']-output['T_res']
#output['rel'] = output['abs']/output['T_ref']
#print(output.iloc[equip,:])
print(output)
output.to_csv('./Cases/' + model_name + '_out.csv')
#print(best_case)

#totals = prob.compute_totals()#of=['T'], wrt=['Spacer5'])
#print(totals)
#check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=True, step=1e-04)
#prob.check_totals(compact_print=True)

#prob.model.list_inputs(print_arrays=True)
