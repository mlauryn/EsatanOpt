import os
import openmdao.api as om
import pandas as pd
import numpy as np
from Pre_process import parse_vf, parse_cond, inits, conductors, nodes, opticals, idx_dict, parse_ar
from Thermal_MDF_unc import Thermal_MDF_unc

# number of design points
npts = 8
model_name = 'CUBESATT'

# define faces to include in radiative analysis
#face_IDs = list(groups.face_IDs()) # import all nodes?
face_IDs = [
    'outer_surf'
    #'Box:outer',
    #'Panel_outer:solar_cells',
    #'Panel_inner:solar_cells',
    #'Panel_body:solar_cells',
    #'Panel_inner: back',
    #'Panel_outer:back',
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

# indices for solar cells
""" sc_idx = sum([idx[face_IDs] for face_IDs in [
    'Panel_outer:solar_cells',
    'Panel_inner:solar_cells',
    'Panel_body:solar_cells'
    ]], []) """

# initial values for some input variables
#prob['cr'][sc_idx] = 1.0
prob['alp_r'][list(idx['outer_surf'])] = 0.9
prob['QI'] = np.tile(QI_init, npts)
prob['phi'] = np.arange(0.,96.,12.)
#prob['QI'][[-4]] = 0.3
#prob['dist'][1] = 3.


prob.run_model()
#prob.run_driver()
#print(prob['T'][1:,:]-273.15)
df = pd.DataFrame(data=prob['T'][1:,:]-273.15, columns=prob['phi'])
df.T.to_pickle('./Cases/' + model_name + '.pkl')
#output['T_res1'] = prob['T'][1:,0]-273.15
#output['T_res2'] = prob['T'][1:,1]-273.15
#output['abs'] = output['T_ref']-output['T_res']
#output['rel'] = output['abs']/output['T_ref']
print(df)

