from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder
from Pre_process import nodes, parse_cond, idx_dict
import numpy as np
from fnmatch import fnmatch
import os
import pandas as pd
from time import time

from MAT_group import MAT_MDP_Group

Sun_dist = [1., 2.75, 1.] # distance from Sun in AU

npts = len(Sun_dist)

MS_model = 'MAT'
RU_model = 'RU_v5_1'

# define faces to include in radiative analysis
MS_geom = [
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

RU_geom =[
'Box:outer',
'Panel_outer:solar_cells',
'Panel_inner:solar_cells',
#'Panel_body:solar_cells',
'Panel_inner: back',
'Panel_outer:back',
'thruster_outer',
#'reel_box_inner',
#'reel_outer'
]

# Instantiate s/c models
model = MAT_MDP_Group(Sun_dist=Sun_dist, MS_model=MS_model, MS_geom=MS_geom, RU_model=RU_model, RU_geom=RU_geom)

fpath = os.path.dirname(os.path.realpath(__file__))
model_dir = fpath + '/Esatan_models/'
RU_data = os.path.join(model_dir, RU_model, 'nodes_output.csv')
MS_data = os.path.join(model_dir, MS_model, 'nodes_output.csv')

# import user-defined conductors
RU_cond = os.path.join(model_dir, RU_model, 'cond_report.txt')
MS_cond = os.path.join(model_dir, MS_model, 'cond_report.txt')

RU_user_cond = parse_cond(RU_cond)
MS_user_cond = parse_cond(MS_cond)

#import node data
nn, groups, output_RU = nodes(data=RU_data)
RU_nodes = sum([groups[group] for group in RU_geom], [])

# user defined node groups
groups.update({'radiator':[158]})# extra node to disipate heat in structure

# global indices for components with controlled heat disipation
RU_equip = sum([groups[syst] for syst in [
    'Prop',
    'obc',
    'radiator'
    ]], [])
# global indices into flattened array
RU_glob = np.arange(0,(nn+1)*npts).reshape((nn+1,npts))

# local index dictionary of radiative nodes_list
RU_local = idx_dict(sorted(RU_nodes), groups)

nn, groups, output_MS = nodes(data=MS_data)

MS_nodes = sum([groups[group] for group in MS_geom], [])

# global indices for components with controlled heat disipation
MS_equip = sum([groups[syst] for syst in [
    'Propulsion_bot',
    'PCB',
    'Battery',
    'Instrument_inner',
    'Esail_bot',
    'TRx',
    'AOCS',
    'RW_Z',
    # radiators
    'SP_Xplus_upr',
    'SP_Xminus_upr',
    'SP_Yplus_upr',
    'SP_Yminus_upr',
    'SP_Zplus_upr',
    'SP_Zminus_upr', 
    ]], [])

# global indices into flattened array
MS_glob = np.arange(0,(nn+1)*npts).reshape((nn+1,npts))

# local index dictionary of radiative nodes_list
MS_local = idx_dict(sorted(MS_nodes), groups)

####################################
# otpimization problem formulation #
####################################

# design variables

# Add broadcast parameters
model.add_design_var('bp.spinAngle', lower=0., upper=45.)

# RU var
for i in range(1, 6):
    for k in range(1, 5):
        model.add_design_var('RU.Spacer{}_{}'.format(i,k), lower=0.25, upper=237.)
#model.add_design_var('RU.Body_panel', lower=0.004, upper=.1)
model.add_design_var('RU.Hinge_inner_1', lower=0.02, upper=.1)
model.add_design_var('RU.Hinge_inner_2', lower=0.02, upper=.1)
model.add_design_var('RU.Hinge_outer_1', lower=0.02, upper=.1)
model.add_design_var('RU.Hinge_outer_2', lower=0.02, upper=.1)
#model.add_design_var('RU.cr', lower=0.0, upper=1., indices=list(idx['Panel_body:solar_cells'])) # only body solar cells are selected here
model.add_design_var('RU.alp_r', lower=0.07, upper=0.94, indices=list(RU_local['Box:outer'])) # optimize absorbptivity for structure
model.add_design_var('RU.Box:outer', lower=0.02, upper=0.94) # optimize emissivity of structure
model.add_design_var('RU.Panel_outer:back', lower=0.02, upper=0.94) # optimize emissivity of solar array back surface
model.add_design_var('RU.Panel_inner: back', lower=0.02, upper=0.94) # optimize emissivity of solar array back surface
model.add_design_var('RU.QI', lower = 0., upper=7., indices=(RU_glob[RU_equip,:]).ravel())

# MS variables

# seperate hinges because they have different bounds
MS_conn = [cond for cond in MS_user_cond if fnmatch(cond['cond_name'], 'Hinge*')==False]
MS_hinges = [cond for cond in MS_user_cond if fnmatch(cond['cond_name'], 'Hinge*')]

for cond in MS_conn:
    model.add_design_var('MS.'+cond['cond_name'], lower=0.0, upper=10. ) # all conductors except hinges
for cond in MS_hinges:
    model.add_design_var('MS.'+cond['cond_name'], lower=0.02, upper=.1 ) # solar array hinges

model.add_design_var('MS.SP_Xplus_upr', lower=0.02, upper=0.94)
model.add_design_var('MS.SP_Xminus_upr', lower=0.02, upper=0.94)
model.add_design_var('MS.SP_Yplus_upr', lower=0.02, upper=0.94)
model.add_design_var('MS.SP_Yminus_upr', lower=0.02, upper=0.94)
model.add_design_var('MS.SP_Zplus_upr', lower=0.02, upper=0.94)
model.add_design_var('MS.SP_Zminus_upr', lower=0.02, upper=0.94)
model.add_design_var('MS.Esail_bot', lower=0.02, upper=0.94)
model.add_design_var('MS.Propulsion_top', lower=0.02, upper=0.94)
model.add_design_var('MS.QI', lower = 0., upper=10., indices=(MS_glob[MS_equip,:]).ravel())

radiator_surf = sum([MS_local[surf] for surf in [
    'SP_Xplus_upr',
    'SP_Xminus_upr',
    'SP_Yplus_upr',
    'SP_Yminus_upr',
    'SP_Zplus_upr',
    'SP_Zminus_upr']], [])

model.add_design_var('MS.alp_r', lower=0.07, upper=0.94, indices=radiator_surf)

# RU constraints
model.add_constraint('RU.bat_lwr.KS', upper=0.0)
model.add_constraint('RU.bat_upr.KS', upper=0.0)
model.add_constraint('RU.prop_upr.KS', upper=0.0)
model.add_constraint('RU.prop_lwr.KS', upper=0.0)
model.add_constraint('RU.obc_pwr.KS', upper=0.0)
model.add_constraint('RU.prop_pwr.KS', upper=0.0)

# MS constraints
model.add_constraint('MS.bat_lwr.KS', upper=0.0)
model.add_constraint('MS.bat_upr.KS', upper=0.0)
model.add_constraint('MS.prop_upr.KS', upper=0.0)
model.add_constraint('MS.prop_lwr.KS', upper=0.0)
model.add_constraint('MS.equip_upr.KS', upper=0.0)
model.add_constraint('MS.equip_lwr.KS', upper=0.0)

model.add_constraint('MS.obc_pwr.KS', upper=0.0)
model.add_constraint('MS.prop_pwr.KS', upper=0.0)
model.add_constraint('MS.ins_pwr.KS', upper=0.0)
model.add_constraint('MS.aocs_pwr.KS', upper=0.0)
model.add_constraint('MS.es_pwr.KS', upper=0.0)

# spinAngle constraint at cruise
#model.add_constraint('bp.spinAngle', equals=33., indices=[0])

# objective function
model.add_objective('MS.P_trx')

prob = Problem(model=model)
prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer']='SLSQP'
prob.driver.options['disp'] = True
prob.driver.options['maxiter'] = 300
prob.driver.options['tol'] = 1.0e-4
#prob.driver.opt_settings['minimizer_kwargs'] = {"method": "SLSQP", "jac": True}
#prob.driver.opt_settings['stepsize'] = 0.01
prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']
prob.driver.add_recorder(SqliteRecorder('./Cases/MAT_mdp.sql'))

prob.setup(check=True)
#prob['bp.spinAngle'] = 30.

run_start = time()
#prob.run_model()
prob.run_driver()
run_time = time() - run_start
print('Run Time:', run_time, 's')
#print('Memory Usage:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.0, 'MB')

temp_RU = pd.DataFrame(data=prob['RU.T'][1:,:]-273.15, index=output_RU.index)
temp_RU['label'] = output_RU[0]
temp_RU.to_csv('./Cases/MAT_mdp_RU.csv')
temp_MS = pd.DataFrame(data=prob['MS.T'][1:,:]-273.15, index=output_MS.index)
temp_MS['label'] = output_MS[0]
temp_MS.to_csv('./Cases/MAT_mdp_MS.csv')

print(temp_RU, temp_MS)