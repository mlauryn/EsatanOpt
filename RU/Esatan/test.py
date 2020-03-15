from scipy.optimize import fsolve
import numpy
def heat_steady(T, GL, GR, QS, QI):
    return(GL.dot(T)-GR*(T**4)+QS+QI)

QI = numpy.array([0,1])
GR = numpy.array([0,0])
GL = numpy.array([-1,1])
QS = numpy.array([0,0])

T0 = numpy.array([273,273])
T = fsolve(heat_steady, T0, args=(GL, GR, QS, QI))
print(T-273)