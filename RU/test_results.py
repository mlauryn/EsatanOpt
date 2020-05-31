import os
import pandas as pd
import matplotlib.pyplot as plt

model_name = 'CUBESATT'

fpath = os.path.dirname(os.path.realpath(__file__))
model_dir = fpath + '/Esatan_models/' + model_name

env='99999'
inact='99998'
ref_df = pd.read_csv(model_dir+'/temp_output.csv', header=1)
ref_df = ref_df.filter(regex = 'T(?!{})(?!{})(\d+)'.format(env, inact))
#temp.columns = temp.columns.str.extract(r'T(\d+)', expand=False) # keep just node number
#temp.index = df['TIME']
#output = nodes.append(temp).T

res_data = pd.read_pickle('./Cases/' + model_name + '.pkl')

ref_data = ref_df.iloc[1:9,:]
ref_data.index = res_data.index
res_data.columns = ref_data.columns

#print(res_data, ref_data)

ax = plt.gca()
ref_data.plot(kind='line', y='T1', ax=ax)
res_data.plot(kind='line', y='T1', ax=ax)

plt.show()

