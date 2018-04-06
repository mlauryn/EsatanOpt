from openmdao.api import Problem, Group, IndepVarComp, ExternalCode
class ParaboloidExternalCode(ExternalCode):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.input_file = 'paraboloid_input.dat'
        self.output_file = 'paraboloid_output.dat'

        # providing these is optional; the component will verify that any input
        # files exist before execution and that the output files exist after.
        self.options['external_input_files'] = [self.input_file,]
        self.options['external_output_files'] = [self.output_file,]

        self.options['command'] = [
            'python', 'extcode_paraboloid.py', self.input_file, self.output_file
        ]

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        # generate the input file for the paraboloid external code
        with open(self.input_file, 'w') as input_file:
            input_file.write('%.16f\n%.16f\n' % (x,y))

        # the parent compute function actually runs the external code
        super(ParaboloidExternalCode, self).compute(inputs, outputs)

        # parse the output file from the external code and set the value of f_xy
        with open(self.output_file, 'r') as output_file:
            f_xy = float(output_file.read())

        outputs['f_xy'] = f_xy

prob = Problem()
model = prob.model

# create and connect inputs
model.add_subsystem('p1', IndepVarComp('x', 3.0))
model.add_subsystem('p2', IndepVarComp('y', -4.0))
model.add_subsystem('p', ParaboloidExternalCode())

model.connect('p1.x', 'p.x')
model.connect('p2.y', 'p.y')

# run the ExternalCode Component
prob.setup()
prob.run_model()

# print the output
print(prob['p.f_xy'])