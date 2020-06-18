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

case_file = 'RU_v5_3_case1.sql'
model_name = 'RU_v5_3'

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
    'Box:outer' : 0.1061,
    'SolarArrays' : 0.4625,
    #'Panel_outer:solar_cells' : 0.89,
    #'Panel_inner:solar_cells' : 0.89,
    #'Panel_body:solar_cells',
    #'Panel_inner: back' : 0.035,
    #'Panel_outer:back' : 0.035,
    'thruster' : 0.03,
    #'reel_box',
    #'reel'
    }


fpath = os.path.dirname(os.path.realpath(__file__))
model_dir = fpath + '/Esatan_models/' + model_name
data = model_dir+'/nodes_output.csv'

nn, groups, output, area = nodes(data=data)
print('number of nodes', nn)
rad_nodes = sum([groups[group] for group in geom], [])

# import user-defined conductors
MS_cond = os.path.join(model_dir, 'cond_report.txt')
MS_user_cond = parse_cond(MS_cond)

# index dictionary of radiative nodes_list
idx = idx_dict(sorted(rad_nodes), groups)


res = pd.DataFrame(index=['Box:outer', 'SolarArrays'], columns=['eps', 'alp'])

eps = [dvs[name] for name in res.index]
res.loc['Box:outer','alp'] = np.mean(dvs['alp_r'][idx['Box:outer']])
res['eps'] = eps

print(res)

