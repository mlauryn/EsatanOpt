"""
Plots objective and constraint histories from the recorded data in 'data.sql'.
"""
from __future__ import print_function

from six.moves import range

import numpy as np
from matplotlib import pylab

from openmdao.api import CaseReader

# load cases from recording database
cr = CaseReader('ru_mdp.sql')
#cases = cr.get_cases('driver')
cases = cr.list_cases('driver')
case = cr.get_case(cases[0])

num_cases = len(cases)
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
    data = cr.get_case(ic).outputs
    X[ic] = data['obj']

    for i in range(n_con):
        con[i] = sum(data[constraints[i]]) # sum inside equality constr if any

    Y[ic] = sum(con)
    Z[ic, :] = con

# generate plots
pylab.figure()

pylab.subplot(311)
pylab.title('Power Output, W')
pylab.plot(X, 'b')

pylab.subplot(312)
pylab.title('Sum of Constraints')
pylab.plot([0, len(Y)], [0, 0], 'k--', marker='o')
pylab.plot(Y, 'k')

pylab.subplot(313)
pylab.title('Max of Constraints')
pylab.plot([0, len(Z)], [0, 0], 'k--')
for i in range(n_con):
    pylab.plot(Z[:, i], marker='o', label=constraints[i])

pylab.legend(loc='best')

pylab.show()