"""
Plots objective and constraint histories from the recorded data in case file.
"""

import numpy as np
from matplotlib import pylab
import re
#import matplotlib.transforms as transforms
from openmdao.api import CaseReader
from plot_size import set_size

case_file = 'MAT_v2_8_case4.sql'

# load cases from recording database
cr = CaseReader('./Cases/'+case_file)
#cases = cr.get_cases('driver')
cases = cr.list_cases('driver')

# extract specific run?
#opt_run = [case for case in cases if re.search(r'Opt_run3', case)]
#cases = opt_run

case = cr.get_case(cases[0])

# objective function name
obj = 'P_trx'

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

# plot style
pylab.style.use('thesis')
pylab.figure(figsize=set_size('thesis', subplots=(2,1)))
pylab.subplots_adjust(left=0.25, hspace=0.25)

# plot objective

pylab.subplot(211)
pylab.plot(-X, '-')
pylab.xlim(0, num_cases)
pylab.ylim(top=10.0)
pylab.xlabel('Function evaluations')
pylab.ylabel('Transmitter Power, W')
pylab.axhline(y=maximum, color="red", label='maximum')
#trans = transforms.blended_transform_factory(
#    pylab.get_yticklabels()[0].get_transform(), pylab.transData)
pylab.text(5,maximum, "{:3.2f}".format(maximum[0]), color="red",  ha="left", va="bottom",) #transform=trans,)
pylab.legend()


# Sum of constraints
pylab.subplot(212)
pylab.plot([0, len(Z)], [0, 0], 'k--')
pylab.plot(Y, '-')
pylab.xlim(0,num_cases)
pylab.ylim(-10000.,7000.)
pylab.xlabel('Function evaluations')
pylab.ylabel('Sum of Constraints')
pylab.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# all constraints
""" pylab.subplot(213)
pylab.plot([0, len(Z)], [0, 0], 'k--')
for i in range(n_con):
    pylab.plot(Z[:, i], markersize=4, label=constraints[i])

pylab.legend(loc='upper right')
pylab.xlim(0,num_cases)
pylab.ylim(-10000.,7000.)
pylab.xlabel('Function evaluations')
pylab.ylabel('Violation of Constraints') """

#pylab.show()
pylab.show()

