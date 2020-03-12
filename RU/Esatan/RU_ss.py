import numpy as np

n = 13 #number of nodes


# Material properties from bulk 'Al_7075T6' 
k_Al_7075T6 = 130.000;  Cp_Al_7075T6 = 960.000;  Dens_Al_7075T6 = 2810.00;  
#
# Material properties from bulk 'PCB_L6' 
k_PCB_L6 = 20.0000;  Cp_PCB_L6 = 900.000;  Dens_PCB_L6 = 1850.00;  

GL = np.zeros((n+1,n+1))

#Conductors (copy from esatan .d file)
GL[1,2] = 0.0200000; # from conductor Hinge_middle
GL[1,5] = 0.0200000; # from conductor Box_to_BodyPanel
GL[2,3] = 0.0200000; # from conductor Hinge_outer
GL[4,5] = 1.0 / ((1.0 / (0.00197 * k_Al_7075T6)) + (1.0 / (0.00182 * k_Al_7075T6))); # from conductive interface ci_4
GL[4,6] = 1.0 / ((1.0 / (0.00203 * k_Al_7075T6)) + (1.0 / (0.00185 * k_Al_7075T6))); # from conductive interface ci_10
GL[4,7] = 1.0 / ((1.0 / (0.00197 * k_Al_7075T6)) + (1.0 / (0.00182 * k_Al_7075T6))); # from conductive interface ci_6
GL[4,8] = 1.0 / ((1.0 / (0.00203 * k_Al_7075T6)) + (1.0 / (0.00185 * k_Al_7075T6))); # from conductive interface ci_7
GL[4,13] = 0.00837500; # from conductor Spacer5
GL[5,6] = 1.0 / ((1.0 / (0.00219 * k_Al_7075T6)) + (1.0 / (0.00217 * k_Al_7075T6))); # from conductive interface ci_11
GL[5,8] = 1.0 / ((1.0 / (0.00219 * k_Al_7075T6)) + (1.0 / (0.00217 * k_Al_7075T6))); # from conductive interface ci_8
GL[5,9] = 1.0 / ((1.0 / (0.00182 * k_Al_7075T6)) + (1.0 / (0.00197 * k_Al_7075T6))); # from conductive interface ci_2
GL[6,7] = 1.0 / ((1.0 / (0.00217 * k_Al_7075T6)) + (1.0 / (0.00219 * k_Al_7075T6))); # from conductive interface ci_9
GL[6,9] = 1.0 / ((1.0 / (0.00185 * k_Al_7075T6)) + (1.0 / (0.00203 * k_Al_7075T6))); # from conductive interface ci_12
GL[7,8] = 1.0 / ((1.0 / (0.00219 * k_Al_7075T6)) + (1.0 / (0.00217 * k_Al_7075T6))); # from conductive interface ci_5
GL[7,9] = 1.0 / ((1.0 / (0.00182 * k_Al_7075T6)) + (1.0 / (0.00197 * k_Al_7075T6))); # from conductive interface ci_3
GL[8,9] = 1.0 / ((1.0 / (0.00185 * k_Al_7075T6)) + (1.0 / (0.00203 * k_Al_7075T6))); # from conductive interface ci_1
GL[9,10] = 0.00737000; # from conductor Spacer1
GL[10,11] = 0.349030; # from conductor Spacer2
GL[11,12] = 0.262190; # from conductor Spacer3
GL[12,13] = 7.25000E-005; # from conductor Spacer4

#remove header row and column, as esatan base starts from 1
GL = GL[1:,1:]

#define diagonal elements as negative of all node conductor couplings (sinks)
diag = -np.sum(GL, 1)

di = np.diag_indices(n)
GL[di] = diag

#make GL matrix symetrical
i_lower = np.tril_indices(n, -1)
GL[i_lower] = GL.T[i_lower]

#define GRs to deep space
""" GR = []
GR[0] = 0.00651463
GR[2] = 0.00616895
GR[3] = 0.00623000
GR[4] = 0.000637237
GR[6] = 0.000667766
GR[7] = 0.000690340
GR[8] = 0.000546717
GR[9] = 0.000637237 """

print(GR)

import pandas as pd
print(pd.DataFrame(GL))
