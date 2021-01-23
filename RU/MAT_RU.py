import os
import openmdao.api as om
import numpy as np
from RU import RemoteUnit
from Pre_process import nodes, inits, idx_dict
from time import time
import pandas as pd

# number of design points
npts = 2

# model version name
model_name = 'RU_v5_6'

# define faces to include in radiative analysis
#geom = list(groups.keys()) # import all nodes?
geom = {
    #'Box:outer' : 0.1061,
    'SolarArrays' : 0.51,
    'SideX': 0.94,
    'SideX_': 0.94,
    'SideY': 0.94,
    'SideY_': 0.94,
    'SideZ': 0.94,
    'SideZ_': 0.94,
    #'thruster' : 0.94,
    #'reel_box',
    #'reel'
    }

fpath = os.path.dirname(os.path.realpath(__file__))
model_dir = fpath + '/Esatan_models/' + model_name
data = model_dir+'/nodes_output.csv'

nn, groups, output, area = nodes(data=data)
QI_init, QS_init = inits(data=data)
rad_nodes = sum([groups[group] for group in geom], [])

# user defined node groups
# groups.update({'radiator':[158]})# extra node to disipate heat in structure

# global indices for components with controlled heat disipation
equip = sum([groups[syst] for syst in [
    'Prop',
    'obc',
    'SolarArrays'
    ]], [])

# global indices into flattened array
flat_indices = np.arange(0,(nn+1)*npts).reshape((nn+1,npts))

# local index dictionary of radiative nodes_list
idx = idx_dict(sorted(rad_nodes), groups)

# remote unit model group instance
model = RemoteUnit(npts=npts, labels=geom, model=model_name)

####################################
# otpimization problem formulation #
####################################

# design variables
for i in range(1, 6):
    for k in range(1, 5):
        model.add_design_var('Spacer{}_{}'.format(i,k), lower=0.25, upper=237.)

model.add_design_var('Hinge_inner_1', lower=0.02, upper=.1)
model.add_design_var('Hinge_inner_2', lower=0.02, upper=.1)
model.add_design_var('Hinge_outer_1', lower=0.02, upper=.1)
model.add_design_var('Hinge_outer_2', lower=0.02, upper=.1)
model.add_design_var('Hinge_screen_1', lower=0.02, upper=.1)
model.add_design_var('Hinge_screen_2', lower=0.02, upper=.1)
model.add_design_var('SideX', lower=0.02, upper=0.94)
model.add_design_var('SideX_', lower=0.02, upper=0.94)
model.add_design_var('SideY', lower=0.02, upper=0.94)
model.add_design_var('SideY_', lower=0.02, upper=0.94)
model.add_design_var('SideZ', lower=0.02, upper=0.94)
model.add_design_var('SideZ_', lower=0.02, upper=0.94)
model.add_design_var('SolarArrays', lower=0.51, upper=0.915)
model.add_design_var('QI', lower = 0., upper=7., indices=(flat_indices[equip,:]).ravel())
model.add_design_var('phi', lower=0., upper=30.)

radiator_surf = sum([idx[surf] for surf in [
    'SideX',
    'SideX_',
    'SideY',
    'SideY_',
    'SideZ',
    'SideZ_',
    #'thruster'
    ]], [])

model.add_design_var('alp_r', lower=0.07, upper=0.94, indices=radiator_surf) # optimize absorbptivity for structure

# constraints
model.add_constraint('bat_lwr.KS', upper=0.0)
model.add_constraint('bat_upr.KS', upper=0.0)
model.add_constraint('prop_upr.KS', upper=0.0)
model.add_constraint('prop_lwr.KS', upper=0.0)
model.add_constraint('obc_pwr.KS', upper=0.0)
model.add_constraint('prop_pwr.KS', upper=0.0)

# objective function
model.add_objective('T_margin')
#model.add_objective('P_prop')

prob = om.Problem(model=model)
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer']='SLSQP'
prob.driver.options['disp'] = True
prob.driver.options['maxiter'] = 200
prob.driver.options['tol'] = 1.0e-4
prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']
prob.driver.add_recorder(om.SqliteRecorder('./Cases/'+ model_name +'_case1.sql'))

prob.setup(check=True)

#prob['phi'] = [0, 30.]
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

""" output['T_res'] = prob['T'][1:,0]-273.15
#output['T_res2'] = prob['T'][1:,1]-273.15
#output['abs'] = output['T_ref']-output['T_res']
#output['rel'] = output['abs']/output['T_ref']
#print(output.iloc[[68,95],:])

output.to_csv('./Cases/' + model_name + '_out.csv') """

temp_RU = pd.DataFrame(data=prob['T'][1:,:]-273.15, index=output.index)
temp_RU['label'] = output[0]
temp_RU.to_csv('./Cases/MAT_RU_out_v6.csv')

#print(best_case)

#totals = prob.compute_totals()#of=['T'], wrt=['Spacer5'])
#print(totals)
#check_partials_data = prob.check_partials(compact_print=True, includes='Cond', show_only_incorrect=False, step=1e-04)
#prob.check_totals(wrt='phi', compact_print=True)

#prob.model.list_inputs(print_arrays=True)
