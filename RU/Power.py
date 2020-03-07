class Power(ExplicitComponent):
    """
    Computes the power output of remote unit solar arrays.
    """
def setup(self):
    self.add_input('q_s', val=1365.)
    self.add_input('length', val=0.1)
    self.add_input('tBPanel', val=1.)
    self.add_input('tDPanel', val=1.)
    self.add_output('Power', val=1.)
    self.add_output('eff', val=1.)
    self.declare_partials('*', '*')
def compute(self, inputs, outputs):
    """solar cell data from: http://www.azurspace.com/images/0005979-01-00_DB_4G32C_Advanced.pdf"""
    T0 = 25. #reference temperature
    eff0 = .318 #efficiency at ref temp
    Vmp0 = 3025. # Vmp in mV at ref temp
    Imp0 = 433.5 # Imp in mA at ref temp
    #cell temperature gradients
    dVmp_dT = -8.6
    dImp_dT = 0.03
    deff_dT = dVmp_dT*dImp_dT/(Vmp0*Imp0)
    eff = eff0 + deff_dT*(T0-inputs['tBPanel'])

    width = 0.115
    area = inputs['length'] * width
    
    Vmp1 = Vmp0 + dVmp_dT * inputs['tBPanel']
    Imp1 = Imp0 + dImp_dT * inputs['tBPanel']

    Vmp1 = Vmp0 + dVmp_dT * inputs['tDPanel']
    Imp1 = Imp0 + dImp_dT * inputs['tDPanel']





