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
ANALYSIS_CASE optimization;
DEFINE_ANALYSIS_CASE (
    analysis_case = optimization,
    working_directory = "{path}");
RUN_ANALYSIS(
    analysis_case = optimization,
    file = "RU_cold.d");
END_MODEL'''.format(path=os.getcwd()))
file.close()

file = open("RU_cold.bat", "w")
file.write('''esrde<RU_cold.ere''')
file.close()

class RU_cold(ExternalCode):
    def setup(self):
        self.add_input('batH', val=0.1)
        self.add_input('propH', val=0.1)
        self.add_input('eps-Y', val=0.5)
        self.add_output('tBat', val=0.0)
        self.add_output('tProp', val=0.0)
        self.add_output('tMain', val=0.0)

        self.input_file = 'RU_cold.d'
        self.output_file = 'RU_cold.out'

        # providing these is optional; the component will verify that any input
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
        eps100 = inputs['eps-Y']
        qi300 = inputs['propH']
        qi10 = inputs['batH']
        eps100 = '{0},'.format(float(eps100)) 
        qi300 = '{0}'.format(float(qi300)) 
        qi10 = '{0}'.format(float(qi10))
        qi20 = qi10 

        # generate the input file for esatan thermal analysis
        generator = InputFileGenerator()
        generator.set_template_file('RU_cold_template.txt')
        generator.set_generated_file('RU_cold.d')
        generator.mark_anchor("$NODES")
        generator.transfer_var(eps100, 53, 9)
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
indeps.add_output('batH', 0.01) 
indeps.add_output('propH', 0.01) 
indeps.add_output('eps-Y', 0.5) 

model.add_subsystem('esatan', RU_cold(), promotes=['*'])

#objective function is total heater power
model.add_subsystem('obj', ExecComp('Power = batH + propH'), promotes=['*'])

# find optimal solution with SciPy optimize
prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'COBYLA'
prob.driver.options['disp'] = False
prob.driver.opt_settings = {'maxiter': 100, 'tol':1e-02, 'catol':1e-01}

prob.model.add_design_var('batH', lower=0.0, upper=1.0)
prob.model.add_design_var('propH', lower=0.0, upper=1.0)
prob.model.add_design_var('eps-Y', lower=0.02, upper=1.0)

prob.model.add_objective('Power')

#constraint for payload electronics max temperature
prob.model.add_constraint('tBat', lower=0.0)
prob.model.add_constraint('tProp', lower=-10.0)
prob.model.add_constraint('tMain', lower=-40.0)

# run the ExternalCode Component
prob.setup(check=True, mode='fwd')
#prob.run_model()
prob.run_driver()

# print the output temperature, minimum radiator length and area
print(prob['Power'])
print(prob['tBat'])
print(prob['tProp'])
print(prob['tMain'])
#print(prob['indeps.RadLen'])
#print(prob['obj.A'])