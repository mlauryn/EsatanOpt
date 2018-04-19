from openmdao.api import Problem, Group, IndepVarComp, ExternalCode
from openmdao.utils.file_wrap import InputFileGenerator, FileParser

class Radiator(ExternalCode):
    def setup(self):
        self.add_input('RadLen', val=0.4)

        self.add_output('T_max', val=0.0)

        self.input_file = 'radiator.d'
        self.output_file = 'radiator.out'

        # providing these is optional; the component will verify that any input
        # files exist before execution and that the output files exist after.
        self.options['external_input_files'] = [self.input_file,]
        self.options['external_output_files'] = [self.output_file,]

        self.options['timeout'] = 10.0
        self.options['fail_hard'] = False
        self.options['command'] = [
            'C:/Users/Laurynas/EsatanOpt/Radiator/radiator.bat']

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
model.add_subsystem('p1', IndepVarComp('RadLen', 0.4))
model.add_subsystem('p2', Radiator())

model.connect('p1.RadLen', 'p2.RadLen')

# run the ExternalCode Component
prob.setup()
prob.run_model()

# print the output
print(prob['p2.T_max'])