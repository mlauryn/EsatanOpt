class SolarCell(ExplicitComponent):
    """
    Computes the dependancy of remote unit solar cell efficiency with temperature.
    """
def setup(self):
    self.add_input('tBPanel', val=1.)
    self.add_input('tDPanel', val=1.)
    self.add_output('eff', val=1.)
    self.declare_partials('*', '*')
def compute(self, inputs, outputs):
    """solar cell data from: http://www.azurspace.com/images/0005979-01-00_DB_4G32C_Advanced.pdf"""
    T0 = 25. #reference temperature
    eff0 = .318 #efficiency at ref temp
    Vmp = 3025. # Vmp in mV at ref temp
    Imp = 433.5 # Imp in mA at ref temp
    #cell temperature gradients
    dVmp_dT = -8.6
    dImp_dT = 0.03
    deff_dT = (dVmp_dT*Imp + dImp_dT*Vmp)/(Vmp*Imp)
    delta_T = (inputs['tBPanel']+inputs['tDPanel'])/2-T0 #taken as average of both arrays
    outputs['eff'] = eff0 * (1 + deff_dT * delta_T)

def compute_partials(self, inputs, partials):
    partials['eff', 'tBPanel'] = eff0 * deff_dT * 0.5
    partials['eff', 'tDPanel'] = eff0 * deff_dT * 0.5



