#Python script for optimization of MAT remote unit thermal model @cold analysis case
import os, time
from openmdao.api import Problem, Group, IndepVarComp, ExternalCode, ScipyOptimizeDriver, SimpleGADriver, ExecComp, ExplicitComponent
from openmdao.utils.file_wrap import InputFileGenerator, FileParser

#generate RU_cold batch mode run files
#note: location of RU_cold command line files should be added to your system path variable  
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
        self.add_input('eps', val=0.02)
        self.add_input('alp', val=0.19)
        self.add_input('GlBat1', val=0.4)
        self.add_input('GlBat2', val=0.4)
        self.add_input('GlMain', val=0.4)
        self.add_input('GlProp', val=0.4)
        self.add_input('GlTether', val=0.4)
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
        self.add_output('tTether', val=0.0)

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
        alp = inputs['alp']
        eps = inputs['eps']
        GL11_200 = inputs['GlBat1']
        GL21_200 = inputs['GlBat2']
        GL107_200 = inputs['GlMain']
        GL111_300 = inputs['GlProp']
        GL103_400 = inputs['GlTether']
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

        alp = '{0},'.format(float(alp)) 
        eps = '{0},'.format(float(eps)) 
        GL11_200 = '{0};'.format(float(GL11_200))
        GL21_200 = '{0};'.format(float(GL21_200))
        GL107_200 = '{0};'.format(float(GL107_200))
        GL111_300 = '{0};'.format(float(GL111_300))
        GL103_400 = '{0};'.format(float(GL103_400))
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

        # generate the input file for RU_cold thermal analysis
        generator = InputFileGenerator()
        generator.set_template_file('RU_cold_template.txt')
        generator.set_generated_file('RU_cold.d')
        generator.mark_anchor("$NODES")
        generator.transfer_var(alp, 53, 6)
        generator.transfer_var(eps, 53, 9)
        generator.transfer_var(alp, 61, 6)
        generator.transfer_var(eps, 61, 9)
        generator.transfer_var(alp, 69, 6)
        generator.transfer_var(eps, 69, 9)
        generator.transfer_var(alp, 77, 6)
        generator.transfer_var(eps, 77, 9)
        generator.transfer_var(alp, 85, 6)
        generator.transfer_var(eps, 85, 9)
        generator.transfer_var(alp, 93, 6)
        generator.transfer_var(eps, 93, 9)
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
        generator.transfer_var(GL103_400, 44, 3)
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
        tTether = parser.transfer_var(36, 3)

        outputs['tBat'] = tBat
        outputs['tMain'] = tMain
        outputs['tProp'] = tProp
        outputs['tTether'] = tTether

class PenaltyFunction(ExplicitComponent):
    """
    Evaluates temperature constraint violation as 
    component actual and required temperature difference 
    if constraint is violated and 0 if not violated
    """
    def setup(self):
        self.add_input('tBat_c',val=0.0)
        self.add_input('tProp_c',val=0.0)
        self.add_input('tMain_c',val=0.0)
        self.add_output('Penalty',val=0.0)

    def compute(self, inputs, outputs):
        tBat_c = inputs('tBat_c')
        tProp_c = inputs('tProp_c')
        tMain_c = inputs('tMain_c')
        tBat_min = 10.0 
        tProp_min = 0.0 
        tMain_min = -30.0
        deltatBat_c = tBat_min - tBat_c 
        deltatProp_c = tProp_min - tProp_c 
        deltatMain_c = tMain_min - tMain_c 

        outputs['penalty'] = max(0, deltatBat_c)+max(0, deltatProp_c)+max(0, deltatMain_c)

prob = Problem()
model = prob.model

# create and connect inputs and outputs
indeps = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
indeps.add_output('batH', val=0.2)
indeps.add_output('propH', val=0.2)
""" indeps.add_output('eps', val=0.2)
indeps.add_output('alp', val=0.4)
indeps.add_output('GlBat1', val=0.4)
indeps.add_output('GlBat2', val=0.4)
indeps.add_output('GlMain', val=0.04)
indeps.add_output('GlProp', val=0.04)
indeps.add_output('GlTether', val=0.04)
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
indeps.add_output('ci12', val=0.4)  """


model.add_subsystem('RU_cold', RU_cold(), promotes_inputs=['*'], promotes_outputs=[('tBat','tBat_c'), 
                    ('tMain','tMain_c'), ('tProp','tProp_c')])
model.add_subsystem('PenaltyFunction', PenaltyFunction(), promotes=['*'])


#objective function is total heater power plus penalty of violating temperature constraints
model.add_subsystem('obj', ExecComp('obj_p = batH + propH + Penalty'), promotes=['*'])


#run the ExternalCode Component once and record initial values
prob.setup(check=True)
prob.run_model()

tBat_1 =  prob['tBat_c']
tProp_1 =  prob['tProp_c']
tMain_1 =  prob['tMain_c']

""" prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer']='SLSQP'
prob.driver.options['disp'] = True
prob.driver.opt_settings = {'eps': 1.0e-6, 'ftol':1e-04,} """
# find optimal solution with simple GA driver
prob.driver = SimpleGADriver()
prob.driver.options['bits'] = {'batH':6, 'propH':6}
""" prob.driver.options['bits'] = {'batH':6, 'propH':6, 'eps': 5, 'alp': 5, 'GlBat1': 5, 'GlBat2':5, 'GlMain':8, 'GlProp':8, 'GlTether':8,
    'ci1':6, 'ci2':6, 'ci3':6, 'ci4':6, 'ci5':6, 'ci6':6, 'ci7':6, 'ci8':6, 'ci9':6, 'ci10':6, 'ci11':6, 'ci12':6} """
prob.driver.options['max_gen'] = 20
#prob.driver.options['run_parallel'] = 'true'
prob.driver.options['debug_print'] = ['desvars']

prob.model.add_design_var('batH', lower = 0.0, upper=1.0)
prob.model.add_design_var('propH', lower = 0.0, upper=1.0)
""" prob.model.add_design_var('eps', lower = 0.02, upper=0.8)
prob.model.add_design_var('alp', lower = 0.23, upper=0.48)
prob.model.add_design_var('GlBat1', lower = 0.4, upper=26.0)
prob.model.add_design_var('GlBat2', lower = 0.4, upper=26.0)
prob.model.add_design_var('GlMain', lower = 0.004, upper=1.0)
prob.model.add_design_var('GlProp', lower = 0.004, upper=1.0)
prob.model.add_design_var('GlTether', lower = 0.004, upper=1.0)
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
prob.model.add_design_var('ci12', lower = 0.015, upper=0.084) """

prob.model.add_objective('obj_p')

#constraint for  temperatures
""" prob.model.add_constraint('tBat_c', lower=0.0, upper = 45.0)
prob.model.add_constraint('tProp_c', lower=-10.0, upper = 80.0)
prob.model.add_constraint('tMain_c', lower=-40.0, upper = 85.0) """

#Run optimization
tStart = time.time()
prob.setup(check=True)
prob.run_driver()

#Record final temperatures
tBat_2 =  prob['tBat_c']
tProp_2 =  prob['tProp_c']
tMain_2 =  prob['tMain_c']

print("Temperatures before optimization:, tBat_1={}, tProp_1={}, tMain_1={}".format(tBat_1, tProp_1, tMain_1)) 
print("Temperatures after optimization:, tBat_2={}, tProp_2={}, tMain_2={}".format(tBat_2, tProp_2, tMain_2))
""" print("Final design variables: eps={}, alp={}, GlBat1={}, GlBat2={}, GlMain={}, GlProp={}, GlTether={}, ci1={}, ci2={}, ci3={}, ci4={}, ci5={}, ci6={}, ci7={}, ci8={}, ci9={}, ci10={}, ci11={}, ci12={}".format (prob['eps'], prob['alp'], prob['GlBat1'], prob['GlBat2'], prob['GlMain'], prob['GlProp'], prob['GlTether'],
prob['ci1'], prob['ci2'], prob['ci3'], prob['ci4'], prob['ci5'], prob['ci6'], prob['ci7'], prob['ci8'], prob['ci9'], prob['ci10'], prob['ci11'], prob['ci12'])) """
print("Optimization run time in minutes:", (time.time()-tStart)/60) 