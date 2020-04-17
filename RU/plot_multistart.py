"""
Plots objective and constraint histories from the recorded data in 'data.sql'.
"""
import numpy as np
from pylab import *
from openmdao.api import CaseReader
import re
samples = [5,10,20, 30,40]
model_name = 'RU_v4_detail'

local_min = []
obj_values = []
glob_min = 0.0

for num in samples:
    
    file_name = model_name + '_mstart_{num}.sql'.format(num=num)
    
    # load cases from recording database
    cr = CaseReader(file_name)
    #cases = cr.get_cases('driver')
    cases = cr.list_cases('driver')
    num_cases = len(cases)
    if num_cases == 0:
        print('No data yet...')
        quit()
    else:
        print('# cases:', num_cases)

    objs = []
    for i in range(num):
        opt_run = [case for case in cases if re.search(r'Opt_run'+str(i), case)]
        last_case = cr.get_case(opt_run[-1])
        obj = last_case.get_objectives()
        if obj['obj'] < glob_min:
            glob_min = obj['obj']
            best_case = last_case
            best_sample = file_name
        objs.append(obj['obj'])
        
    local_min.append(min(objs))
    obj_values.extend(objs)
print('Global minimum ', min(local_min), 'at ', best_case.name, 'in sample', best_sample)

# generate plots
""" plot(samples, local_min)
xlabel('# samples')
ylabel('global minimum')
show() """

plot(obj_values, 'r.')
show()