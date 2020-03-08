import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS, FullFactorial

""" 'length', lower = 0.0, upper=0.254
'eff', lower = 0.25, upper=0.32
'eps', lower = 0.02, upper=0.8
'P_ht', lower = 0.0, upper=1.0
'r_bat', lower = 0.0, upper=1.0
'GlMain', lower = 0.004, upper=1.0
'GlProp', lower = 0.004, upper=1.0
'GlPanel', lower = 0.004, upper=1.0 """

xlimits = np.array([[0.0, 0.254], [0.25, 0.32], [0.02, 0.8], [0.0, 1.0], [0.0, 1.0], [150., 1365.], [1, 250], [1, 250], [1, 250]])
sampling = LHS(xlimits=xlimits, criterion='ese')

num = 200
x = sampling(num)


np.savetxt('./Samples/RU_LHsample[ese]_n=200.csv', x, delimiter=',', header = 'length, eff, eps, ht_gain, r_bat, q_s, R_m, R_p, R_s',
comments = '')

plt.plot(x[:, 2], x[:, 7], "o")
plt.xlabel("x")
plt.ylabel("y")
plt.show()