import openmdao.api as om
import numpy as np

from RU_MDP import RU_MDP


from Pre_process import parse_vf, parse_cond, inits, conductors, nodes, opticals, idx_dict

npts = 2

nn, groups = nodes()
GL_init, GR_init = conductors(nn=nn, data='cond_RU_v4_base_cc.csv')

cond_data = parse_cond(filepath='links_RU_v4_base.txt') 
optprop = parse_vf(filepath='vf_RU_v4_base.txt')

#keys = list(groups.keys()) # import all nodes?
keys = ['Box', 'Panel_outer', 'Panel_inner', 'Panel_body'] # define faces to include in radiative analysis
faces = opticals(groups, keys, optprop)
nodes = sum([groups[group] for group in keys], [])
print(nodes)

# index dictionary or radiative nodes
idx = idx_dict(nodes, groups)

model = RU_MDP(nn=nn, npts=npts, faces=faces, model='RU_v4_base', conductors=cond_data, GL_init=GL_init, GR_init=GR_init)

prob = om.Problem(model=model)

model.add_design_var('Spacer5', lower=0.25, upper=237.)
model.add_design_var('Spacer1', lower=0.25, upper=237.)
model.add_design_var('Hinge_inner', lower=0.004, upper=.1)
model.add_design_var('Hinge_middle', lower=0.02, upper=.1)
model.add_design_var('Hinge_outer', lower=0.02, upper=.1)

model.add_design_var('cr', lower=0.0, upper=1., indices=list(idx['Panel_body'])) # only body solar cells are selected here
model.add_design_var('alp_r', lower=0.07, upper=0.94, indices=[8]) # optimize absorbptivity for structure
model.add_design_var('Box', lower=0.02, upper=0.94) # optimize emissivity of structure
model.add_design_var('QI', lower = 0.25, upper=7., indices=[-1, -2, -7, -8, -10])
model.add_design_var('phi', lower=0., upper=90.)

#model.add_constraint('T', lower=0.+273, upper=45.+273, indices=[-1, -2])
#model.add_constraint('power_bal.KS', upper=0.0)
model.add_constraint('bat_lwr.KS', upper=0.0)
model.add_constraint('bat_upr.KS', upper=0.0)
model.add_constraint('prop_upr.KS', upper=0.0)
model.add_constraint('prop_lwr.KS', upper=0.0)
""" model.add_constraint('obc_pwr.KS', upper=0.0)
model.add_constraint('prop_pwr.KS', upper=0.0) """


model.add_objective('obj')
model.linear_solver = om.DirectSolver()
model.linear_solver.options['assemble_jac'] = False

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer']='SLSQP'
prob.driver.options['disp'] = True
prob.driver.options['maxiter'] = 70
prob.driver.options['tol'] = 1.0e-4
#prob.driver.opt_settings['minimizer_kwargs'] = {"method": "SLSQP", "jac": True}
#prob.driver.opt_settings['stepsize'] = 0.01
prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']
prob.driver.add_recorder(om.SqliteRecorder("ru_mdp.sql"))

prob.setup(check=True)

# indices for solar cells
sc_idx = sum([idx[keys] for keys in ['Panel_outer', 'Panel_inner', 'Panel_body']], [])

# initial values for solar cells and radiators

prob['cr'][sc_idx] = 1.0
prob['alp_r'][list(idx['Box'])] = 0.5
prob['QI'][[-1]] = 0.2
prob['QI'][[-4]] = 0.3

""" cr = om.CaseReader('thermal_mdp.sql')
cases = cr.list_cases('driver')
num_cases = len(cases)
print(num_cases)

# Load the last case written
last_case = cr.get_case(cases[num_cases-1])
prob.load_case(last_case) """

#prob.run_model()
prob.run_driver()
print(prob['T']-273.)

#totals = prob.compute_totals(of=['T'], wrt=['Spacer5'])
#print(totals)
#check_partials_data = prob.check_partials(compact_print=True, show_only_incorrect=True, step=1e-04)
#prob.check_totals(compact_print=True)

#print(prob['P_dis'])
#print(prob['P_in'], prob['P_out'])
#print(prob['QS_c'], prob['QS_r'])

#prob.model.list_inputs(print_arrays=True)