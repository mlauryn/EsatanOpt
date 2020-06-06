import pandas as pd
import matplotlib.pyplot as plt
from plot_size import set_size
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np


plt.style.use('thesis')

fd = pd.read_csv('Cases/converg_fdm.csv', header=None, names=['nn', 't', 'rel'])
fem = pd.read_csv('Cases/converg_fem.csv', header=None, names=['nn', 't', 'rel'])

# make smooth curves
""" x = np.linspace(fem['nn'].min(), fem['nn'].max(), 200)
fd_spl = make_interp_spline(fd['nn'].to_numpy(), fd['t'].to_numpy(), k=1)
fem_spl = make_interp_spline(fem['nn'], fem['t'], k=1)

t_fd = fd_spl(x)
t_fem = fem_spl(x)
 """

# generate plots
fig, ax1 = plt.subplots(figsize=set_size('thesis'))
fig.subplots_adjust(left=0.15, bottom=0.15)


ax1.plot(fd['nn'], fd['t'], '-o', label='FDM')
ax1.plot(fem['nn'], fem['t'], '-s', label='FEM')
ax1.set(xlabel='Number of nodes', xticks=fd['nn'], ylabel=r'$T_1, ^\circ C$')

ax2 = ax1.twinx()
ax2.plot(fd['nn'], fd['rel'], '-o', label='FDM')
ax2.plot(fem['nn'], fem['rel'], '-s', label='FEM')
ax2.set(ylabel='rel error')
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax1.grid(False)

plt.legend()

plt.show()
