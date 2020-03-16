#Simple component that computes the dependancy of remote unit solar cell efficiency with temperature.

import openmdao.api as om 

class SolarCell(om.ExplicitComponent):  
    def setup(self):
        self.add_input('tBPanel', val=1.)
        self.add_input('tDPanel', val=1.)
        self.add_output('eff', val=1.)
        self.declare_partials('*', '*')
    def compute(self, inputs, outputs):
        """solar cell data from:https://www.e3s-conferences.org/articles/e3sconf/pdf/2017/04/e3sconf_espc2017_03011.pdf"""
        T0 = 28. #reference temperature
        eff0 = .285 #efficiency at ref temp
        T1 = -150.
        eff1 = 0.335

        delta_T = (inputs['tBPanel']+inputs['tDPanel'])/2-T0 #taken as average of both arrays

        deff_dT = (eff1 - eff0) / (T1 - T0)
        
        outputs['eff'] = eff0 + deff_dT * delta_T

    def compute_partials(self, inputs, partials):
        
        T0 = 28. #reference temperature
        eff0 = .285 #efficiency at ref temp
        T1 = -150.
        eff1 = 0.335
        deff_dT = (eff1 - eff0) / (T1 - T0)
        
        partials['eff', 'tBPanel'] = deff_dT * 0.5
        partials['eff', 'tDPanel'] = deff_dT * 0.5

if __name__ == "__main__":
    prob = om.Problem()
    model = prob.model
    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('tBPanel', val=100)
    indeps.add_output('tDPanel', val=100)

    model.add_subsystem('sc', SolarCell(), promotes=['*'])

    prob.setup(check=True)
    prob.run_model()

    print(prob['tBPanel'], prob['tDPanel'], prob['eff'])



