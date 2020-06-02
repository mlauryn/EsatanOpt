import os
import pandas as pd
import matplotlib.pyplot as plt
from plot_size import set_size

plt.style.use('thesis')

model_name = 'CUBESATT'

fpath = os.path.dirname(os.path.realpath(__file__))
model_dir = fpath + '/Esatan_models/' + model_name

env='99999'
inact='99998'
ref_df = pd.read_csv(model_dir+'/temp_output.csv', header=1)
Tref = ref_df.filter(regex = 'T(?!{})(?!{})(\d+)'.format(env, inact))
Qref = ref_df.filter(regex = 'QS(?!{})(?!{})(\d+)'.format(env, inact))
#temp.columns = temp.columns.str.extract(r'T(\d+)', expand=False) # keep just node number
#temp.index = df['TIME']

Tres = pd.read_pickle('./Cases/' + model_name + '_t.pkl')
Qres = pd.read_pickle('./Cases/' + model_name + '_q.pkl')

Tref = Tref.iloc[1:9,:]
Qref = Qref.iloc[1:9,:]
Tref.index = Tres.index
Qref.index = Qres.index
#print(Tres, Tref)
#print(Qres, Qref)

# statistics
Qref['total'] = Qref.sum(1)
Qres['total'] = Qres.sum(1)
Trel = Tres[1]/Tref['T1'] - 1.
Qrel = Qres['total']/Qref['total'] - 1.
Tabs = Tres[1]-Tref['T1']
Qabs = Qres['total'] - Qref['total']

# generate plots
fig, axs = plt.subplots(2,2, sharex=True, figsize=set_size('thesis', subplots=(3,2)))
fig.subplots_adjust(wspace=0.45, right=0.97)

Tref['T1'].plot(ax=axs[0,0], marker='o', label='Esatan-TMS',); axs[0,0].set(title='a', ylabel=r'$T1, ^\circ C$')
Tres[1].plot(ax=axs[0,0], ls='--', marker='+', label='OpenMDAO')
Tabs.plot(ax=axs[1,0], marker='+'); axs[1,0].set(title='c', ylabel=r'$T1$ abs error, $^\circ C$')
Qref['total'].plot(ax=axs[0,1], marker='o', label='Esatan-TMS')
Qres['total'].plot(ax=axs[0,1], ls='--', marker='+', label='OpenMDAO'); axs[0,1].set(title='b', ylabel=r'$Q_{tot}, W$')
Qrel.plot(ax=axs[1,1], marker='+'); axs[1,1].set(title='d', ylabel=r'$Q_{tot}$ rel error')

for ax in axs[1,:]:
    ax.set(xlabel=r'$\Phi$, degrees', xticks=Tres.index.values)

axs[1,1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')

#plt.show()
plt.savefig('./Figures/sm_valid.pdf' , format='pdf')
