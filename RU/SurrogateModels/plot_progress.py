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
""" cases = CaseReader('ru_mdp.sql').driver_cases
cases.load_cases() """

num_cases = len(cases)
if num_cases == 0:
    print('No data yet...')
    quit()
else:
    print('# cases:', num_cases)

# determine the # of points (2 constraints per point)
constraints = case.get_constraints().keys()
print(constraints)
n_point = len(constraints) // 2

# collect data into arrays for plotting
X = np.zeros(num_cases)       # obj.val
Y = np.zeros((num_cases, 2))       # sum of constraints
Z = np.zeros((num_cases, 2))  # constraints

for ic in range(num_cases):
    data = cr.get_case(ic).outputs
    X[ic] = -data['length']

    for ip in range(n_point):
        Y[ic,ip] = data['pt%d.tBat' % ip]
        Z[ic,ip] = data['pt%d.tProp' % ip]

# generate plots
pylab.figure()

pylab.subplot(211)
pylab.title('Solar Panel length')
pylab.plot(X, 'b')
#pylab.plot([0, len(X)], [3e4, 3e4], 'k--', marker='o')

""" pylab.subplot(312)
pylab.title('Sum of Constraints')
pylab.plot([0, len(Y)], [0, 0], 'k--', marker='o')
pylab.plot(Y, 'k') """

pylab.subplot(212)
pylab.title('Temperature Constraints')
#pylab.plot([0, len(Z)], [0, 0], 'k--')
pylab.plot(Y[:, 0], 'r', label='tBat_h')
pylab.plot(Y[:, 1], 'r', label='tBat_c')
pylab.plot(Z[:, 0], 'b', label='tProp_h')
pylab.plot(Z[:, 1], 'b', label='tProp_c')

pylab.legend(loc='best')

pylab.show()