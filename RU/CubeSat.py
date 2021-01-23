import openmdao.api as om
import numpy as np 
from GLmtxComp import GLmtxComp
from GRmtxComp import GRmtxComp
from TempComp import TempComp
from Pre_process import nodes, inits, conductors, parse_vf, opticals, parse_ar, parse_cond
import pandas as pd

model_name = 'CUBESAT1'
model_dir = './Esatan_models/'+model_name
n, groups, output, area = nodes(data=model_dir+'/nodes_output.csv')
GL_init, GR_init = conductors(nn=n, data=model_dir+'/cond_output.csv')
QI_init, QS_init = inits(data=model_dir+'/nodes_output.csv')
optprop = parse_vf(filepath=model_dir+'/vf_report.txt')
areas = parse_ar(filepath=model_dir+'/area.txt')

# define faces to include in radiative analysis
keys = [
    'outer_surf',
    #'Box:outer',
    #'Panel_outer:solar_cells',
    #'Panel_inner:solar_cells',
    #'Panel_body',
    #'Panel_inner: back',
    #'Panel_outer:back',
] 

faces = opticals(groups, keys, optprop, areas)
#user_cond = parse_cond(filepath=model_dir + '/cond_report.txt')

model = om.Group()

input_var = model.add_subsystem('input', om.IndepVarComp(), promotes=['*'])
#for cond in user_cond:
#    input_var.add_output(cond['cond_name'], val=cond['values'][0] )
for face in faces:
    input_var.add_output(face['name'], val=face['eps'][0] )
input_var.add_output('GL', val=GL_init)
#input_var.add_output('GR', val=GR_init)
input_var.add_output('QI', val=QI_init)
input_var.add_output('QS', val=QS_init)

#model.add_subsystem('GLmtx', GLmtxComp(n=n, GL_init=GL_init, user_links=user_cond), promotes=['*'])
model.add_subsystem('GRmtx', GRmtxComp(n=n, GR_init=GR_init, faces=faces), promotes=['*'])
model.add_subsystem('TMM', TempComp(n=n), promotes=['*'])
model.add_design_var('outer_surf', lower=0.03, upper=0.95)
model.add_subsystem('obj', om.ExecComp('obj = (T_1 - 293.15)**2'), promotes=['*'])
model.connect('T', 'T_1', src_indices=1)
model.add_objective('obj')

model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
model.nonlinear_solver.options['iprint'] = 2
model.nonlinear_solver.options['maxiter'] = 50
model.linear_solver = om.DirectSolver(assemble_jac=True)
model.options['assembled_jac_type'] = 'csc'

problem = om.Problem(model=model)

problem.driver = om.ScipyOptimizeDriver()
problem.driver.options['optimizer']='SLSQP'
problem.driver.options['disp'] = True
problem.driver.options['maxiter'] = 70
#problem.driver.options['tol'] = 1.0e-4
problem.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']
problem.driver.add_recorder(om.SqliteRecorder('./Cases/'+ model_name +'.sql'))

problem.setup(check=True)
    
#problem.run_model()
problem.run_driver()

output['T_res'] = problem['T'][1:]-273.15
output['abs'] = output['T_ref']-output['T_res']
output['rel'] = output['abs']/output['T_ref']
print(output)
#problem.model.list_inputs(print_arrays=True)
#np.set_printoptions(threshold=np.inf)
#print((np.linalg.norm(problem['GR'])-np.linalg.norm(GR_init))/np.linalg.norm(GR_init))