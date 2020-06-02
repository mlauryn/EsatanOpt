import pandas as pd
import matplotlib.pyplot as plt
from plot_size import set_size
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np


plt.style.use('thesis')

fd = pd.read_csv('Cases/converg_fdm.csv', header=None, names=['nn', 't'])
fem = pd.read_csv('Cases/converg_fem.csv', header=None, names=['nn', 't'])

# make smooth curves
""" x = np.linspace(fem['nn'].min(), fem['nn'].max(), 200)
fd_spl = make_interp_spline(fd['nn'].to_numpy(), fd['t'].to_numpy(), k=1)
fem_spl = make_interp_spline(fem['nn'], fem['t'], k=1)

t_fd = fd_spl(x)
t_fem = fem_spl(x)
 """

# generate plots
fig = plt.figure(figsize=set_size('thesis'))
fig.subplots_adjust(left=0.15, bottom=0.15)


plt.plot(fd['nn'], fd['t'], '-o', label='FDM')
plt.plot(fem['nn'], fem['t'], '-s', label='FEM')
ax = plt.gca()
ax.set(xlabel='Number of nodes', ylabel=r'$T_1, ^\circ C$')
plt.legend()

plt.show()
#plt.savefig('./Figures/sm_valid.pdf' , format='pdf')
