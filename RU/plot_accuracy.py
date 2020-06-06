import os
import pandas as pd
import matplotlib.pyplot as plt
from plot_size import set_size
import numpy as np

env='99999'
inact='99998'
norm = []
alp = [0.01, 0.1, 0.2, 0.4, 0.8, 0.95]

plt.style.use('thesis')
ref_df = pd.read_csv('./Esatan_models/RU_v4_detail/nodes_output.csv', header=1).filter(regex = 'T(?!{})(?!{})(\d+)'.format(env, inact)).T

model_name = 'RU_v4_set'
Tref = pd.DataFrame(index=ref_df.index)
Tres = pd.DataFrame(index=ref_df.index)
fpath = os.path.dirname(os.path.realpath(__file__))
for i in range(1,7):

    model_dir = fpath + '/Esatan_models/' + model_name + str(i) 
    ref_df = pd.read_csv(model_dir+'/nodes_output.csv', header=1).filter(regex = 'T(?!{})(?!{})(\d+)'.format(env, inact)).T
    Tref[i] = ref_df[0]
    res_df = pd.read_pickle('./Cases/' + model_name +  str(i) + '_out.pkl')
    Tres[i] = res_df['T_1']
    norm.append(np.linalg.norm(Tref[i]))
    norm.append(np.linalg.norm(Tres[i])) 

# statistics
norm = np.array(norm).reshape((6,2))
rel = (norm[:,0]-norm[:,1])/norm[:,0]
diff = Tres.set_index(Tref.index).subtract(Tref)

# generate plots
fig, ax1 = plt.subplots(figsize=set_size('thesis'))
fig.subplots_adjust(left=0.15, bottom=0.15)


ax1.plot(alp, diff.max(0), '-o', color='red')
ax1.set_ylabel(r'max abs error, $^\circ C$', color='red')
ax1.set_xlabel(r'$\alpha$')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_xticks(alp)

ax2 = ax1.twinx()
ax2.plot(alp,rel, '-o', color='blue')
ax2.set_ylabel('norm rel error', color='blue')
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax2.tick_params(axis='y', labelcolor='blue')
ax2.grid(False)

plt.show()



