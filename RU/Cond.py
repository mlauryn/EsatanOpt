import openmdao.api as om
import numpy as np
from GLMatrixComp_copy import GLMatrixComp_copy
from TempComp import TempComp

class Cond(om.Group)
    def __init__(self, SF, indices, GL_init, GR_init, n, QS, QI):
            super(MPPT, self).__init__()

            self.SF = SF
            self.indices = indices
            self.GL_init = GL_init
            self.GR_init = GR_init
            self.n = n
            self.QS = QS
            self.QI = QI
    
    def setup(self)
        n_in = len(self.SF)
        n = self.n
        GL_init = self.GL_init
        indices = self.indices
        SF = self.SF
        
        param = om.IndepVarComp()
        for i, idx in enumerate(indices):
            i_name = 'k{}'.format(idx)
            param.add_output(i_name)
        param.add_output(QS, val=self.QS)
        param.add_output(QI, val=self.QI)
        param.add_output(GR, val=self.GR_init)

        self.add_subsystem('param', param, promotes=['*'])
        self.add_subsystem('GLmtx', GLMatrixComp_copy(n=n, GL_init=GL_init, indices=indices, SF=SF), promotes=['*'])
        self.add_subsystem('Temp', TempComp(n=n))

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 20
        self.linear_solver = om.DirectSolver()

if __name__ == "__main__":
    
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
    GL[12,13] = 7.25000e-5; # from conductor Spacer4

    #remove header row and column, as esatan base starts from 1
    GL = GL[1:,1:]

    #make GL matrix symetrical
    i_lower = np.tril_indices(n, -1)
    GL[i_lower] = GL.T[i_lower]

    #define diagonal elements as negative of all node conductor couplings (sinks)
    diag = -np.sum(GL, 1)

    di = np.diag_indices(n)
    GL[di] = diag

    """ import pandas as pd
    print(pd.DataFrame(GL)) """

    #define GRs to deep space
    GR = np.zeros(n+1)

    GR[1] = 0.00651463
    GR[2] = 0.00616895
    GR[3] = 0.00623000
    GR[4] = 0.000637237
    GR[6] = 0.000667766
    GR[7] = 0.000690340
    GR[8] = 0.000546717
    GR[9] = 0.000637237

    GR = GR[1:]

    #esatan does not include stefan-boltzman const
    sigma = 5.670374e-8
    GR = GR*sigma

    #define external solar flux
    QS =np.zeros(n+1)
    QS[1] = 0.681670
    QS[2] = 0.645368
    QS[3] = 0.645368
    QS[4] = 0.000000
    QS[5] = 0.000000
    QS[6] = 0.000000
    QS[7] = 0.000000
    QS[8] = 0.000000
    QS[9] = 0.000000
    QS[10] = 0.000000
    QS[11] = 0.000000
    QS[12] = 0.000000
    QS[13] = 0.000000

    QS = QS[1:]

    #define internal heat disipation
    QI = np.zeros(n+1)
    QI[13] = 0.200000
    QI[10] = 0.300000
    QI = QI[1:]

    model = cond()
    p['GL'] = GL
    p['GR'] = GR
    p['QS'] = QS
    p['QI'] = QI

    #gues values
    p['T'] = -np.ones(n)*50 + 273
    p.run_model()
    print(p['T']-273.15)