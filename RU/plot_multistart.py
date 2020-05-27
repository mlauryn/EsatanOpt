"""
Plots local minima of multiple optimizer runs from the recorded data in 'modelname_mstart_#ofsamples.sql' and finds global minimum.
"""
import numpy as np
from pylab import *
from openmdao.api import CaseReader
import re
samples = [1,5,10,20, 30, 40]
model_name = 'RU_v4_detail'

local_min = []
obj_values = []
glob_min = 0.0

for num in samples:
    
    file_name = './Cases/' + model_name + '_mstart_{num}.sql'.format(num=num)
    
    # load cases from recording database
    cr = CaseReader(file_name)
    #cases = cr.get_cases('driver')
    cases = cr.list_cases('driver')
    
    # determine # of constraints
    case = cr.get_case(cases[0])
    constraints = list(case.get_constraints())
    n_con = len(constraints)

    num_cases = len(cases)
    if num_cases == 0:
        print('No data yet...')
        quit()
    else:
        print('# cases:', num_cases)

    objs = []
    for i in range(num):
        opt_run = [case for case in cases if re.search(r'Opt_run'+str(i), case)]
        # open last iteration of optimizer run
        last_case = cr.get_case(opt_run[-1])

        # check if this is feasible solution
        feasible = True
        con = last_case.get_constraints()
        for k in range(n_con):
            if sum(con[constraints[k]]) > 1e-02: #should be less than constraint tolerance
                feasible = False
                break
        
        if feasible == False:
            continue # continue to next run if this was not feasible
        else:
            # check if this is global minimum
            obj = last_case.get_objectives()
            if obj['obj'] < glob_min:
                glob_min = obj['obj']
                best_case = last_case
                best_sample = file_name
            # add result to list
            objs.append(obj['obj'])
    # add best result to list of local minima    
    local_min.append(glob_min)
    # add all results to global list
    obj_values.extend(objs)
print('Global minimum ', min(local_min), 'at ', best_case.name, 'in sample', best_sample)

# generate plots
y1 = np.array(local_min)
y2 = np.array(obj_values)
subplot(121)
plot(samples, -y1, 'ob-')
#ylim(bottom=0)
xticks(ticks=samples)
xlabel('Number of samples', fontsize=14)
ylabel('Best objective function value ', fontsize=14)
plt.grid(ls='--')

subplot(122)
plot(-y2, 'r.')
xlabel('Number of samples', fontsize=14)
ylabel('Objective function values', fontsize=14)
plt.grid(ls='--')

show()