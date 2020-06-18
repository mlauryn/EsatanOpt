"""
Prints optimization optults.
"""

import numpy as np
from matplotlib import pylab
#import re
#import matplotlib.transforms as transforms
from openmdao.api import CaseReader
from Pre_process import nodes, idx_dict, parse_cond
import pandas as pd
import os

case_file = 'MAT_v2_8_case3.sql'
model_name = 'MAT_v2_8'

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

fpath = os.path.dirname(os.path.realpath(__file__))
model_dir = fpath + '/Esatan_models/' + model_name
data = model_dir+'/nodes_output.csv'

nn, groups, output, area = nodes(data=data)

print('number of nodes', nn)

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

radiators = [
    'SP_Xplus',
    'SP_Xminus',
    'SP_Yplus',
    'SP_Yminus',
    'SP_Zplus',
    'SP_Zminus',
    'Propulsion',
    'Esail',
    'Instrument',
    'BottomPlate'
    ]

equip = [
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
    ]

rad_nodes = sum([groups[group] for group in geom], [])

# import user-defined conductors
MS_cond = os.path.join(model_dir, 'cond_report.txt')
MS_user_cond = parse_cond(MS_cond)

# index dictionary of radiative nodes_list
idx = idx_dict(sorted(rad_nodes), groups)


opt = pd.DataFrame(index=radiators+['SolarArray'] , columns=['eps', 'alp'])
Qdis = pd.DataFrame(index=equip, columns=['cold', 'hot'])

for comp in equip:
    Qdis.loc[comp,'cold'] = np.sum(dvs['QI'][groups[comp], 0])
    Qdis.loc[comp,'hot'] = np.sum(dvs['QI'][groups[comp], 1])

eps = [dvs[name] for name in opt.index]
for var in radiators:
    opt.loc[var,'alp'] = np.mean(dvs['alp_r'][idx[var]])
opt['eps'] = eps

print(opt, Qdis)

