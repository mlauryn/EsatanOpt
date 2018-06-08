#Python script for optimization of MAT remote unit thermal model
import os
from openmdao.api import Problem, Group, IndepVarComp, ExternalCode, ScipyOptimizeDriver, ExecComp
from openmdao.utils.file_wrap import InputFileGenerator, FileParser

#generate esatan batch mode run files
#note: location of esatan command line files should be added to your system path variable  
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('.\\esatan')
file = open("RU_cold.ere", 'w')
file.write('''BEGIN_MODEL RU_cold
ANALYSIS_CASE optimization;'
DEFINE_ANALYSIS_CASE (
    analysis_case = optimization,
    working_directory = "{path}");'
RUN_ANALYSIS(
    analysis_case = optimization,
    file = "RU_cold.d");'
END_MODEL'''.format(path=os.getcwd()))
file.close()

file = open("RU_cold.bat", "w")
file.write('''esrde<RU_cold.ere''')
file.close()

class RU_cold(ExternalCode):
    def setup(self):
        self.add_input('batH', val=0.1)
        self.add_input('propH', val=0.1)
        self.add_input('epsX', val=0.4)
        self.add_input('alpX', val=0.4)
        self.add_input('eps-X', val=0.4)
        self.add_input('alp-X', val=0.4)
        self.add_input('epsY', val=0.4)
        self.add_input('alpY', val=0.4)
        self.add_input('eps-Y', val=0.4)
        self.add_input('alp-Y', val=0.4)
        self.add_input('epsZ', val=0.4)
        self.add_input('alpZ', val=0.4)
        self.add_input('eps-Z', val=0.4)
        self.add_input('alp-Z', val=0.4)
        self.add_input('GlBat1', val=0.4)
        self.add_input('GlBat2', val=0.4)
        self.add_input('GlMain', val=0.4)
        self.add_input('GlProp', val=0.4)
        self.add_input('ci1', val=0.4)
        self.add_input('ci2', val=0.4)
        self.add_input('ci3', val=0.4)
        self.add_input('ci4', val=0.4)
        self.add_input('ci5', val=0.4)
        self.add_input('ci6', val=0.4)
        self.add_input('ci7', val=0.4)
        self.add_input('ci8', val=0.4)
        self.add_input('ci9', val=0.4)
        self.add_input('ci10', val=0.4)
        self.add_input('ci11', val=0.4)
        self.add_input('ci12', val=0.4)
        self.add_output('tBat', val=0.0)
        self.add_output('tProp', val=0.0)
        self.add_output('tMain', val=0.0)

        self.input_file = 'RU_cold.d'
        self.output_file = 'RU_cold.out'

        # providing these is optional;' the component will verify that any input
        # files exist before execution and that the output files exist after.
        #self.options['external_input_files'] = [self.input_file,]
        #self.options['external_output_files'] = [self.output_file,]
        #self.options['poll_delay'] = 10.0
        self.options['timeout'] = 10.0
        self.options['fail_hard'] = False
        self.options['command'] = [
            'RU_cold.bat']
        # this external code does not provide derivatives, use finite difference
        #self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        alp100 = inputs['alpZ']
        eps100 = inputs['epsZ']
        alp102 = inputs['alp-Y']
        eps102 = inputs['eps-Y']
        alp104 = inputs['alpX']
        eps104 = inputs['epsX']
        alp106 = inputs['alpY']
        eps106 = inputs['epsY']
        alp108 = inputs['alp-X']
        eps108 = inputs['eps-X']
        alp110 = inputs['alp-Z']
        eps110 = inputs['eps-Z']
        GL11_200 = inputs['GlBat1']
        GL21_200 = inputs['GlBat2']
        GL107_200 = inputs['GlMain']
        GL111_300 = inputs['GlProp']
        GL107_111 = inputs['ci1']
        GL107_109 = inputs['ci2']
        GL103_109 = inputs['ci3']
        GL101_109 = inputs['ci4']
        GL109_111 = inputs['ci5']
        GL101_103 = inputs['ci6']
        GL103_111 = inputs['ci7']
        GL101_107 = inputs['ci8']
        GL103_105 = inputs['ci9']
        GL101_105 = inputs['ci10']
        GL105_111 = inputs['ci11']
        GL105_107 = inputs['ci12']
        qi300 = inputs['propH']
        qi10 = inputs['batH']

        alp100 = '{0},'.format(float(alp100)) 
        eps100 = '{0},'.format(float(eps100)) 
        alp102 = '{0},'.format(float(alp102)) 
        eps102 = '{0},'.format(float(eps102)) 
        alp104 = '{0},'.format(float(alp104)) 
        eps104 = '{0},'.format(float(eps104)) 
        alp106 = '{0},'.format(float(alp106)) 
        eps106 = '{0},'.format(float(eps106)) 
        alp108 = '{0},'.format(float(alp108)) 
        eps108 = '{0},'.format(float(eps108)) 
        alp110 = '{0},'.format(float(alp110)) 
        eps110 = '{0},'.format(float(eps110))
        GL11_200 = '{0};'.format(float(GL11_200))
        GL21_200 = '{0};'.format(float(GL21_200))
        GL107_200 = '{0};'.format(float(GL107_200))
        GL111_300 = '{0};'.format(float(GL111_300))
        GL107_111 = '{0};'.format(float(GL107_111))
        GL107_109 = '{0};'.format(float(GL107_109))
        GL103_109 = '{0};'.format(float(GL103_109))
        GL101_109 = '{0};'.format(float(GL101_109))
        GL109_111 = '{0};'.format(float(GL109_111))
        GL101_103 = '{0};'.format(float(GL101_103))
        GL103_111 = '{0};'.format(float(GL103_111))
        GL101_107 = '{0};'.format(float(GL101_107))
        GL103_105 = '{0};'.format(float(GL103_105))
        GL101_105 = '{0};'.format(float(GL101_105))
        GL105_111 = '{0};'.format(float(GL105_111))
        GL105_107 = '{0};'.format(float(GL105_107)) 
        qi300 = '{0}'.format(float(qi300)) 
        qi10 = '{0}'.format(float(qi10))
        qi20 = qi10 

        # generate the input file for esatan thermal analysis
        generator = InputFileGenerator()
        generator.set_template_file('RU_cold_template.txt')
        generator.set_generated_file('RU_cold.d')
        generator.mark_anchor("$NODES")
        generator.transfer_var(alp100, 53, 6)
        generator.transfer_var(eps100, 53, 9)
        generator.transfer_var(alp102, 61, 6)
        generator.transfer_var(eps102, 61, 9)
        generator.transfer_var(alp104, 69, 6)
        generator.transfer_var(eps104, 69, 9)
        generator.transfer_var(alp106, 77, 6)
        generator.transfer_var(eps106, 77, 9)
        generator.transfer_var(alp108, 85, 6)
        generator.transfer_var(eps108, 85, 9)
        generator.transfer_var(alp110, 93, 6)
        generator.transfer_var(eps110, 93, 9)
        generator.mark_anchor("Generated conductors")
        generator.transfer_var(GL11_200, 1, 3)  
        generator.transfer_var(GL21_200, 2, 3)
        generator.transfer_var(GL107_200, 3, 3)
        generator.transfer_var(GL111_300, 4, 3)
        generator.transfer_var(GL107_111, 5, 3)
        generator.transfer_var(GL107_109, 6, 3)
        generator.transfer_var(GL103_109, 7, 3)
        generator.transfer_var(GL101_109, 8, 3)
        generator.transfer_var(GL109_111, 9, 3)
        generator.transfer_var(GL101_103, 10, 3)
        generator.transfer_var(GL103_111, 11, 3)
        generator.transfer_var(GL101_107, 12, 3)
        generator.transfer_var(GL103_105, 13, 3)
        generator.transfer_var(GL101_105, 14, 3)
        generator.transfer_var(GL105_111, 15, 3)
        generator.transfer_var(GL105_107, 16, 3)
        generator.mark_anchor("$INITIAL")
        generator.transfer_var(qi300, 7, 3)
        generator.transfer_var(qi10, 10, 3)
        generator.transfer_var(qi20, 13, 3)
        generator.generate()

        # the parent compute function actually runs the external code
        super(RU_cold, self).compute(inputs, outputs)

        # parse the output file from the external code and set the value of T_max
        parser = FileParser()
        parser.set_file('RU_cold.out')
        parser.mark_anchor("+RU_COLD")
        tBat = parser.transfer_var(7, 3)
        tMain = parser.transfer_var(34, 3)
        tProp = parser.transfer_var(35, 3)

        outputs['tBat'] = tBat
        outputs['tMain'] = tMain
        outputs['tProp'] = tProp

prob = Problem()
model = prob.model

# create and connect inputs and outputs
indeps = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
indeps.add_output('batH', val=0.1)
indeps.add_output('propH', val=0.1)
indeps.add_output('epsX', val=0.4)
indeps.add_output('alpX', val=0.4)
indeps.add_output('eps-X', val=0.4)
indeps.add_output('alp-X', val=0.4)
indeps.add_output('epsY', val=0.4)
indeps.add_output('alpY', val=0.4)
indeps.add_output('eps-Y', val=0.4)
indeps.add_output('alp-Y', val=0.4)
indeps.add_output('epsZ', val=0.4)
indeps.add_output('alpZ', val=0.4)
indeps.add_output('eps-Z', val=0.4)
indeps.add_output('alp-Z', val=0.4)
indeps.add_output('GlBat1', val=0.4)
indeps.add_output('GlBat2', val=0.4)
indeps.add_output('GlMain', val=0.4)
indeps.add_output('GlProp', val=0.4)
indeps.add_output('ci1', val=0.4)
indeps.add_output('ci2', val=0.4)
indeps.add_output('ci3', val=0.4)
indeps.add_output('ci4', val=0.4)
indeps.add_output('ci5', val=0.4)
indeps.add_output('ci6', val=0.4)
indeps.add_output('ci7', val=0.4)
indeps.add_output('ci8', val=0.4)
indeps.add_output('ci9', val=0.4)
indeps.add_output('ci10', val=0.4)
indeps.add_output('ci11', val=0.4)
indeps.add_output('ci12', val=0.4) 


model.add_subsystem('esatan', RU_cold(), promotes=['*'])

#objective function is total heater power
model.add_subsystem('obj', ExecComp('Power = batH * 2 + propH'), promotes=['*'])

# find optimal solution with SciPy optimize
prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'COBYLA'
prob.driver.options['disp'] = False
prob.driver.opt_settings = {'maxiter': 300, 'tol':1e-02, 'catol':1e-01}

prob.model.add_design_var('batH', lower = 0.0)
prob.model.add_design_var('propH', lower = 0.0)
prob.model.add_design_var('epsX', lower = 0.02, upper=0.8)
prob.model.add_design_var('alpX', lower = 0.23, upper=0.48)
prob.model.add_design_var('eps-X', lower = 0.02, upper=0.8)
prob.model.add_design_var('alp-X', lower = 0.23, upper=0.48)
prob.model.add_design_var('epsY', lower = 0.02, upper=0.8)
prob.model.add_design_var('alpY', lower = 0.23, upper=0.48)
prob.model.add_design_var('eps-Y', lower = 0.02, upper=0.8)
prob.model.add_design_var('alp-Y', lower = 0.23, upper=0.48)
prob.model.add_design_var('epsZ', lower = 0.02, upper=0.8)
prob.model.add_design_var('alpZ', lower = 0.23, upper=0.48)
prob.model.add_design_var('eps-Z', lower = 0.02, upper=0.8)
prob.model.add_design_var('alp-Z', lower = 0.23, upper=0.48)
prob.model.add_design_var('GlBat1', lower = 0.4, upper=26.0)
prob.model.add_design_var('GlBat2', lower = 0.4, upper=26.0)
prob.model.add_design_var('GlMain', lower = 0.004, upper=1.0)
prob.model.add_design_var('GlProp', lower = 0.004, upper=1.0)
prob.model.add_design_var('ci1', lower = 0.013, upper=0.072)
prob.model.add_design_var('ci2', lower = 0.015, upper=0.084)
prob.model.add_design_var('ci3', lower = 0.015, upper=0.084)
prob.model.add_design_var('ci4', lower = 0.008, upper=0.026)
prob.model.add_design_var('ci5', lower = 0.008, upper=0.026)
prob.model.add_design_var('ci6', lower = 0.013, upper=0.072)
prob.model.add_design_var('ci7', lower = 0.013, upper=0.072)
prob.model.add_design_var('ci8', lower = 0.013, upper=0.072)
prob.model.add_design_var('ci9', lower = 0.015, upper=0.084)
prob.model.add_design_var('ci10', lower = 0.008, upper=0.026)
prob.model.add_design_var('ci11', lower = 0.008, upper=0.026)
prob.model.add_design_var('ci12', lower = 0.015, upper=0.084)

prob.model.add_objective('Power')

#constraint for payload electronics max temperature
prob.model.add_constraint('tBat', lower=0.0)
prob.model.add_constraint('tProp', lower=-10.0)
prob.model.add_constraint('tMain', lower=-40.0)

# run the ExternalCode Component
prob.setup(check=True, mode='fwd')
prob.run_model()
#prob.run_driver()

# print the output temperature, minimum radiator length and area
print(prob['Power'])
print(prob['tBat'])
print(prob['tProp'])
print(prob['tMain'])
#print(prob['indeps.RadLen'])
#print(prob['obj.A'])