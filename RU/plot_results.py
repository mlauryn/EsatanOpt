"""
Plots objective and constraint histories from the recorded data in 'data.sql'.
"""

import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import CaseReader

# load cases from recording database
cr = CaseReader('RU_v4_base_mstart_1.sql')
#cases = cr.get_cases('driver')
cases = cr.list_cases('driver')
initial = cr.get_case(cases[0])
final = cr.get_case(cases[-1])
#num_cases = len(cases)

T1 = initial.outputs['T']-273
T2 = final.outputs['T']-273

N = 4
plt.rcParams.update({'font.size': 14})
ind = np.arange(N) 
width = 0.3       
plt.bar(ind, T1[-4:,0], width, label='Initial temperature at 1 AU',)
plt.bar(ind, T1[-4:,1], width, label='Initial temperature at 3 AU',)
plt.bar(ind + width, T2[-4:,0], width, color='deepskyblue', label='Optimized temperature at 1 AU',)
plt.bar(ind + width, T2[-4:,1], width, color='gold', label='Optimized temperature at 3 AU',)
plt.plot([-0.15, 3.45], [0, 0], 'k--')
plt.grid(axis='y', ls='--')

plt.ylabel('Temperature, $^\circ C$', fontsize=16)
plt.ylim(top=125)
plt.xlabel('Sub-system number', fontsize=16)
#plt.title('Sub-system temperatures before and after optimization')

plt.xticks(ind + width / 2, ('1', '2', '3', '4'))
plt.legend(loc='best')
plt.show()