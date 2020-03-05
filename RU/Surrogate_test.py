#Python script for optimization of MAT remote unit thermal model @cold analysis case using surrogate model
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt 

train = np.loadtxt('RUc_TrainingData_n=20.csv', delimiter=',')
test = np.loadtxt('RUc_TrainingData_n=200.csv', delimiter=',')
ytest = test[:,9]
#print(data)

# train the surrogate
ru_mm = om.MetaModelUnStructuredComp(vec_size=200, default_surrogate=om.KrigingSurrogate())
ru_mm.add_input('eps', val=np.zeros(200), training_data=train[:,0])
ru_mm.add_input('length', val=np.zeros(200), training_data=train[:,1])
ru_mm.add_input('eff', val=np.zeros(200), training_data=train[:,2])
ru_mm.add_input('P_ht', val=np.zeros(200), training_data=train[:,3])
ru_mm.add_input('r_bat', val=np.zeros(200), training_data=train[:,4])
ru_mm.add_input('GlMain', val=np.zeros(200), training_data=train[:,5])
ru_mm.add_input('GlProp', val=np.zeros(200), training_data=train[:,6])
ru_mm.add_input('GlTether', val=np.zeros(200), training_data=train[:,7])
#ru_mm.add_input('GlPanel', val=np.zeros(200), training_data=train[:,8])

ru_mm.add_output('tBat', val=np.zeros(200), training_data=train[:,9])

prob = om.Problem()
model = prob.model

# create and connect inputs and outputs
indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
indeps.add_output('eps', val=test[:,0])
indeps.add_output('length', val=test[:,1])
indeps.add_output('eff', val=test[:,2])
indeps.add_output('P_ht', val=test[:,3])
indeps.add_output('r_bat', val=test[:,4])
#indeps.add_output('alp', val=0.4)
indeps.add_output('GlMain', val=test[:,5])
indeps.add_output('GlProp', val=test[:,6])
indeps.add_output('GlTether', val=test[:,7])
#indeps.add_output('GlPanel', val=test[:,8])

model.add_subsystem('mm', ru_mm, promotes=['*'])


prob.setup(check=True)


prob.run_model()

y = prob['mm.tBat']
print(y)


fig = plt.figure()
plt.plot(ytest, ytest, '-', label='$y_{true}$')
plt.plot(ytest, y, 'r.', label='$\hat{y}$')
       
plt.xlabel('$y_{true}$')
plt.ylabel('$\hat{y}$')
        
plt.legend(loc='upper left')
plt.title('Kriging model: validation of the prediction model')
plt.show()