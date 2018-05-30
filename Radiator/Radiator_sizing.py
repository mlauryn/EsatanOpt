import os
from openmdao.api import Problem, Group, IndepVarComp, ExternalCode, ScipyOptimizeDriver, ExecComp
from openmdao.utils.file_wrap import InputFileGenerator, FileParser

#generate esatan batch mode run files
#note: location of esatan command line files should be added to your system path variable  
file = open("radiator.era", 'w')
file.write('''BEGIN_ADMIN
DELETE_MODEL "radiator";
DELETE_FILE (
    file = "%HOME%\Documents\Esatan\radiator\system\ESATAN-TMS.LCK");
END_ADMIN''')
file.close()

file = open("radiator.ere", 'w')
file.write('''BEGIN_MODEL radiator
ANALYSIS_CASE radiator_sizing;
DEFINE_ANALYSIS_CASE (
    analysis_case = radiator_sizing,
	working_directory = "{path}");
RUN_ANALYSIS(
    analysis_case = radiator_sizing,
    file = "radiator.d");
END_MODEL'''.format(path=os.getcwd()))
file.close()

file = open("radiator.bat", "w")
file.write('''esrde<radiator.ere''')
file.close()

class Radiator(ExternalCode):
    def setup(self):
        self.add_input('RadLen', val=0.4)

        self.add_output('T_max', val=0.0)

        self.input_file = 'radiator.d'
        self.output_file = 'radiator.out'

        # providing these is optional; the component will verify that any input
        # files exist before execution and that the output files exist after.
        #self.options['external_input_files'] = [self.input_file,]
        #self.options['external_output_files'] = [self.output_file,]
        #self.options['poll_delay'] = 10.0
        self.options['timeout'] = 10.0
        self.options['fail_hard'] = False
        self.options['command'] = [
            'radiator.bat']
        # this external code does not provide derivatives, use finite difference
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['RadLen']
        var = '{0};'.format(float(x)) 

        # generate the input file for esatan thermal analysis
        generator = InputFileGenerator()
        generator.set_template_file('radiator_templ.txt')
        generator.set_generated_file('radiator.d')
        generator.mark_anchor("$CONSTANTS")
        generator.transfer_var(var, 4, 3)
        generator.generate()

        # the parent compute function actually runs the external code
        super(Radiator, self).compute(inputs, outputs)

        # parse the output file from the external code and set the value of T_max
        parser = FileParser()
        parser.set_file('radiator.out')
        parser.mark_anchor("+RADIATOR ")
        T_max = parser.transfer_var(5, 6)

        outputs['T_max'] = T_max

prob = Problem()
model = prob.model

# create and connect inputs
indeps = model.add_subsystem('indeps', IndepVarComp())
indeps.add_output('RadLen', 0.4) 
indeps.add_output('width', 0.2) 
model.add_subsystem('esatan', Radiator())

#objective function is radiator area A
model.add_subsystem('obj', ExecComp('A = length * width'))

model.connect('indeps.RadLen', ['esatan.RadLen', 'obj.length'])
model.connect('indeps.width', 'obj.width')

# find optimal solution with SciPy optimize
prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'COBYLA'
prob.driver.options['disp'] = False

prob.model.add_design_var('indeps.RadLen', lower=0.1, upper=0.5)
prob.model.add_objective('obj.A')
#constraint for payload electronics max temperature
prob.model.add_constraint('esatan.T_max', upper=20)

# run the ExternalCode Component
prob.setup(check=True, mode='fwd')
#prob.run_model()
prob.run_driver()

# print the output temperature, minimum radiator length and area
print(prob['esatan.T_max'])
print(prob['indeps.RadLen'])
print(prob['obj.A'])