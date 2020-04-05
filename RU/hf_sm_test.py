import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from smt.surrogate_models import RMTC, RMTB, RBF, IDW, KRG
from parse_hf import parse_hf

filepath = 'radiative_results.txt'
df = parse_hf(filepath)

data = df.loc[0].pivot(index='var', columns='node', values='IS')  

xt = data.index.to_numpy()
yt = data.to_numpy()

xlimits = np.array([[xt[0], xt[-1]]])


sm = RMTB(
    xlimits=xlimits,
    #num_elements=15,
    energy_weight=1e-5,
    regularization_weight=1e-14,
    min_energy=True,
    #order=4,
    num_ctrl_pts=50
)
sm.set_training_values(xt, yt)
sm.train()

num = 90
x = np.linspace(0.0, 90.0, num)
y = sm.predict_values(x)

xd = np.array([0.,45.])
dy_dx = sm.predict_derivatives(xd,0)

plt.plot(xt, yt[:,1], "o")
plt.plot(x, y[:,1])
#plt.plot(x, dy_dx[:,0])
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Training data", "Prediction"])
plt.show()

""" print(np.amin(y))
print(np.amax(y)) """
#print(dy_dx)