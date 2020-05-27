"""
Plots objective and constraint histories from the recorded data in 'data.sql'.
"""

import numpy as np
from matplotlib import pylab
import re
#import matplotlib.transforms as transforms

from openmdao.api import CaseReader

# load cases from recording database
cr = CaseReader('./Cases/RU_v4_detail_mstart_30.sql')
#cases = cr.get_cases('driver')
cases = cr.list_cases('driver')
# filter optimizer run with best result
best_run = [case for case in cases if re.search(r'Opt_run3', case)]
case = cr.get_case(best_run[0])

num_cases = len(best_run)
if num_cases == 0:
    print('No data yet...')
    quit()
else:
    print('# cases:', num_cases)

# determine # of constraints
constraints = list(case.get_constraints())

n_con = len(constraints)

# collect data into arrays for plotting
X = np.zeros(num_cases)       # obj.val
Y = np.zeros(num_cases)       # sum of constraints
Z = np.zeros((num_cases, n_con))  # constraints

con =[1.]* n_con
for ic in range(num_cases):
    data = cr.get_case(best_run[ic]).outputs
    X[ic] = data['obj']

    for i in range(n_con):
        con[i] = sum(data[constraints[i]]) # sum inside equality constr if any

    Y[ic] = sum(con)
    Z[ic, :] = con

maximum = -data['obj']

# generate plots
cons_labels = ['$T_{4, lb}$','$T_{4,ub}$', 'PowerBalance', '$T_{1, lb}$', '$T_{1,ub}$']
pylab.figure()
pylab.rcParams.update({'font.size': 14})

pylab.subplot(211)
pylab.plot(-X, 'ob-')
pylab.xlim(0, 50)
#pylab.ylim(top=1.0)
pylab.xlabel('Function evaluations')
pylab.ylabel('Sub-system #1 Power, W')

pylab.axhline(y=maximum, color="red", label='global maximum')
#trans = transforms.blended_transform_factory(
#    pylab.get_yticklabels()[0].get_transform(), pylab.transData)
pylab.text(5,maximum, "{:3.2f}".format(maximum[0]), color="red",  ha="right", va="bottom",) #transform=trans,)
pylab.legend()

""" pylab.subplot(312)
pylab.title('Sum of Constraints')
pylab.plot([0, len(Y)], [0, 0], 'k--', marker='o')
pylab.plot(Y, 'k') """

pylab.subplot(212)
pylab.plot([0, len(Z)], [0, 0], 'k--')
for i in range(n_con):
    pylab.plot(Z[:, i], marker='o', markersize=4, label=cons_labels[i])

pylab.legend(loc='upper center')
pylab.xlim(0,50)
pylab.xlabel('Function evaluations')
pylab.ylabel('Max of Constraints')

pylab.show()