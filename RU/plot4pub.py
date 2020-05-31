"""
Plots objective and constraint histories from the recorded data in case file.
"""

import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt
import re
#import matplotlib.transforms as transforms
from openmdao.api import CaseReader
from plot_size import set_size

pylab.style.use('thesis')

case_file = 'RU_v5_1.sql'

# load cases from recording database
cr = CaseReader('./Cases/'+case_file)
#cases = cr.get_cases('driver')
cases = cr.list_cases('driver')

# extract specific run?
#opt_run = [case for case in cases if re.search(r'Opt_run3', case)]
#cases = opt_run

case = cr.get_case(cases[0])

# objective function name
obj = 'P_prop'

num_cases = len(cases)
if num_cases == 0:
    print('No data yet...')
    quit()
else:
    print('# cases:', num_cases)

# determine # of constraints
constraints = list(case.get_constraints()) # get all constraints?
#constraints = ['bat_lwr.KS', 'bat_upr.KS', 'prop_lwr.KS', 'prop_upr.KS', ]
n_con = len(constraints)

# collect data into arrays for plotting
X = np.zeros(num_cases)       # obj.val
Y = np.zeros(num_cases)       # sum of constraints
Z = np.zeros((num_cases, n_con))  # constraints

con =[1.]* n_con
for ic in range(num_cases):
    data = cr.get_case(cases[ic]).outputs
    X[ic] = data[obj]

    for i in range(n_con):
        con[i] = sum(data[constraints[i]]) # sum inside equality constr if any

    Y[ic] = sum(con)
    Z[ic, :] = con

maximum = -data[obj]

#########################
# generate plots

cons_labels = ['$T_{obc, lb}$','$T_{obc, ub}$', '$T_{prop, lb}$', '$T_{prop,ub}$', '$P_{obc}$', '$P_{bal}$']

fig, axs = plt.subplots(2, sharex=True, figsize=set_size('thesis', subplots=(2,1)))
ax1, ax2 = axs

ax1.plot(-X, 'ob-')
ax1.axhline(y=maximum, color="red", label='obj value')
ax1.text(5,maximum, "{:3.2f}".format(maximum[0]), color="red",  ha="left", va="bottom",)
ax1.set_ylim(top=1.0)
ax1.set_ylabel('Propulsion Power, W')
ax1.legend()

ax2.plot([0, len(Z)], [0, 0], 'k--')
for i in range(n_con):
    ax2.plot(Z[:, i], marker='o', markersize=4, label=cons_labels[i])
ax2.legend(loc='upper right', ncol=2)
ax2.set(xlabel='Function evaluations', ylabel='Violation of Constraints')

#plt.show()
plt.savefig('./Figures/opt_progress.pdf' , format='pdf')
