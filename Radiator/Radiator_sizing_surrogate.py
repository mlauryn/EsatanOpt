import os
from openmdao.api import Problem, Group, IndepVarComp, ExternalCode, ScipyOptimizeDriver, ExecComp
import openmdao.api as om
from openmdao.utils.file_wrap import InputFileGenerator, FileParser
import numpy as np

#generate esatan batch mode run files
#note: location of esatan command line files should be added to your system path variable  

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
        #self.declare_partials(of='*', wrt='*', method='fd')

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

prob1 = Problem()
model = prob1.model
model.add_subsystem('esatan', Radiator(), promotes=['*'])
model.add_subsystem('DOEX', om.IndepVarComp('RadLen', 0.), promotes=['*'])

model.add_design_var('RadLen', lower=0.1, upper=0.5)
model.add_objective('T_max')

prob1.driver = om.DOEDriver(om.UniformGenerator(num_samples=5))
prob1.driver.add_recorder(om.SqliteRecorder("cases.sql"))

prob1.setup()
prob1.run_driver()
prob1.cleanup()

cr = om.CaseReader("cases.sql")
cases = cr.list_cases('driver')

values = []
for case in cases:
    outputs = cr.get_case(case).outputs
    values.append((outputs['RadLen'], outputs['T_max']))

print("\n".join(["RadLen: %5.2f, T_max: %6.2f" % xf for xf in values]))

metamod = om.MetaModelUnStructuredComp()
metamod.add_input('x', 0.)
metamod.add_output('y', 0., surrogate=om.KrigingSurrogate())
# train the surrogate
values = np.array(values)
metamod.options['train:x'] = values[:,0]
metamod.options['train:y'] = values[:,1]

# create and connect inputs
prob2 = om.Problem()
model = prob2.model

indeps = model.add_subsystem('indeps', IndepVarComp())
indeps.add_output('RadLen', 0.4) 
indeps.add_output('width', 0.2) 
model.add_subsystem('obj', ExecComp('A = length * width'))
model.add_subsystem('mm', metamod)
model.connect('indeps.RadLen', ['mm.x', 'obj.length'])
model.connect('indeps.width', 'obj.width')

# find optimal solution with SciPy optimize
prob2.driver = ScipyOptimizeDriver()
prob2.driver.options['optimizer'] = 'SLSQP'
#prob2.driver.opt_settings = {'eps': 1.0e-12, 'ftol':1e-04}

prob2.model.add_design_var('indeps.RadLen', lower=0.1, upper=0.5)
prob2.model.add_objective('obj.A')
#constraint for payload electronics max temperature
prob2.model.add_constraint('mm.y', upper=20)

prob2.setup(check=True)
#prob2.run_model()
prob2.run_driver()

# print the output temperature, minimum radiator length and area
print(prob2['mm.y'])
print(prob2['indeps.RadLen'])
print(prob2['obj.A'])








