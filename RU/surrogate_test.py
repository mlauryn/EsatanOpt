#Python script for validating surrogate model
import numpy as np
import matplotlib.pyplot as plt
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS
from smt.utils import compute_rms_error 

ndim = 7

train = np.loadtxt('./TrainingData/RUc_TrainingData[ese]_n=50.csv', delimiter=',')
test = np.loadtxt('./TrainingData/RUc_TrainingData[ese]_n=50.csv', delimiter=',')
xtest, ytest = test[:,:ndim], test[:,ndim]
xt, yt = train[:,:ndim], train[:,ndim:]

# The variable 'theta0' is a list of length ndim.
theta = [0.17675797, 0.0329642, 0.00175843, 0.0328348, 0.00039516, 0.08729705,
 0.00094059, 0.00018145, 0.04470183]
t = KRG(theta0=[1e-2]*ndim,print_prediction = False)
t.set_training_values(xt,yt[:,0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print('Kriging,  err: '+ str(compute_rms_error(t,xtest,ytest)))

fig = plt.figure()
plt.plot(ytest, ytest, '-', label='$y_{true}$')
plt.plot(ytest, y, 'r.', label='$\hat{y}$')
       
plt.xlabel('$y_{true}$')
plt.ylabel('$\hat{y}$')
        
plt.legend(loc='upper left')
plt.title('Kriging model: validation of the prediction model')
plt.show()

# Value of theta
print("theta values",  t.optimal_theta)