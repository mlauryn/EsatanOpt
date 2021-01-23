"""
Prints optimization results.
"""

import numpy as np
from matplotlib import pylab
#import re
#import matplotlib.transforms as transforms
from openmdao.api import CaseReader
from Pre_process import nodes, idx_dict, parse_cond
import pandas as pd
import os

case_file = 'RU_v5_6_case1.sql'
model_name = 'RU_v5_6'

# load cases from recording database
cr = CaseReader('./Cases/'+case_file)
cases = cr.list_cases('driver')

# extract specific run?
#opt_run = [case for case in cases if re.search(r'Opt_run3', case)]
#cases = opt_run

case = cr.get_case(cases[-1])

objs = case.get_objectives()
#cons = case.get_constraints()
dvs = case.get_design_vars(use_indices=False)

geom = {
    #'Box:outer' : 0.1061,
    'SolarArrays' : 0.51,
    'SideX': 0.1061,
    'SideX_': 0.1061,
    'SideY': 0.1061,
    'SideY_': 0.1061,
    'SideZ': 0.1061,
    'SideZ_': 0.1061,
    'thruster' : 0.03,
    #'reel_box',
    #'reel'
    }

radiators = [
    'SideX',
    'SideX_',
    'SideY',
    'SideY_',
    'SideZ',
    'SideZ_',
    #'thruster'
    ]

equip = [
    'SolarArrays',
    'obc',
    'Prop'
    ]


fpath = os.path.dirname(os.path.realpath(__file__))
model_dir = fpath + '/Esatan_models/' + model_name
data = model_dir+'/nodes_output.csv'

nn, groups, output, area = nodes(data=data)
print('number of nodes', nn)
rad_nodes = sum([groups[group] for group in geom], [])

# index dictionary of radiative nodes_list
idx = idx_dict(sorted(rad_nodes), groups)

opt = pd.DataFrame(index=radiators+['SolarArrays'] , columns=['eps', 'alp'])
Qdis = pd.DataFrame(index=equip, columns=['cold', 'hot'])

for comp in equip:
    Qdis.loc[comp,'cold'] = np.sum(dvs['QI'][groups[comp], 0])
    Qdis.loc[comp,'hot'] = np.sum(dvs['QI'][groups[comp], 1])

eps = [dvs[name] for name in opt.index]
for var in radiators:
    opt.loc[var,'alp'] = np.mean(dvs['alp_r'][idx[var]])
opt['eps'] = eps

print(opt, '\n', Qdis)

for i in range(1, 6):
    for k in range(1, 5):
        cond = 'Spacer{}_{}'.format(i,k)
        print(cond, dvs[cond])
print(
    'Hinge_inner_1', dvs['Hinge_inner_1'],
    'Hinge_inner_2', dvs['Hinge_inner_2'],
    'Hinge_outer_1', dvs['Hinge_outer_1'],
    'Hinge_outer_2', dvs['Hinge_outer_2'],
    'Hinge_screen_1', dvs['Hinge_screen_1'],
    'Hinge_screen_2', dvs['Hinge_screen_2'])