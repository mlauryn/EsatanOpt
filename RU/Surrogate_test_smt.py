#Python script for optimization of MAT remote unit thermal model @cold analysis case using surrogate model
import numpy as np
import matplotlib.pyplot as plt
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS
from smt.utils import compute_rms_error 

ndim = 9

train = np.loadtxt('RUc_TrainingData_n=100.csv', delimiter=',')
test = np.loadtxt('RUc_TrainingData_n=200.csv', delimiter=',')
xtest, ytest = test[:,:9], test[:,9]
xt, yt = train[:,:9], train[:,9:]

# The variable 'theta0' is a list of length ndim.
theta = [2.01391340e-01, 1.00896469e-02, 4.45720318e-03, 1.17642950e-02,
 4.82131544e-03, 4.61226497e-01, 1.86280940e-04, 3.03366523e-04,
 4.76241459e-03]
t = KRG(theta0=theta,print_prediction = False)
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