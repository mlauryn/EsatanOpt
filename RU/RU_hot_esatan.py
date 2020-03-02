#External code to run Remote Unit hot case in Esatan
import os, time
import openmdao.api as om
from openmdao.utils.file_wrap import InputFileGenerator, FileParser

#generate RU_cold batch mode run files
#note: location of RU_cold command line files should be added to your system path variable  
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('.\\esatan')
file = open("RU_hot.ere", 'w')
file.write('''BEGIN_MODEL RU_hot
ANALYSIS_CASE optimization;'
DEFINE_ANALYSIS_CASE (
    analysis_case = optimization,
    working_directory = "{path}");'
RUN_ANALYSIS(
    analysis_case = optimization,
    file = "RU_hot.d");'
END_MODEL'''.format(path=os.getcwd()))
file.close()

file = open("RU_hot.bat", "w")
file.write('''esrde<RU_hot.ere''')
file.close()

class RU_hot(om.ExternalCode):
    def setup(self):
        self.add_input('length', val=0.1)
        self.add_input('P_un', val=0.4)
        self.add_input('eps', val=0.4)
        self.add_input('alp', val=0.4)
        self.add_input('GlMain', val=0.4)
        self.add_input('GlProp', val=0.4)
        self.add_input('GlTether', val=0.4)
        self.add_input('GlPanel', val=0.4)
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
        self.add_output('tBPanel', val=0.0)
        self.add_output('tDPanel', val=0.0)

        self.input_file = 'RU_hot.d'
        self.output_file = 'RU_hot.out'

        # providing these is optional;' the component will verify that any input
        # files exist before execution and that the output files exist after.
        self.options['external_input_files'] = [self.input_file,]
        self.options['external_output_files'] = [self.output_file,]
        #self.options['poll_delay'] = 10.0
        self.options['timeout'] = 10.0
        self.options['fail_hard'] = False
        self.options['command'] = [
            'RU_hot.bat']
        # this external code does not provide derivatives, use finite difference
        #self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        length = inputs['length']
        alp = inputs['alp']
        eps = inputs['eps']
        GL107_200 = inputs['GlMain']
        GL111_300 = inputs['GlProp']
        GL103_400 = inputs['GlTether']
        GL102_601 = inputs['GlPanel']
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
        P_un = inputs['P_un']

        length = '{0};'.format(float(length))
        P_un = '{0};'.format(float(P_un))
        alp = '{0},'.format(float(alp)) 
        eps = '{0},'.format(float(eps)) 
        GL107_200 = '{0};'.format(float(GL107_200))
        GL111_300 = '{0};'.format(float(GL111_300))
        GL103_400 = '{0};'.format(float(GL103_400))
        GL102_601 = '{0};'.format(float(GL102_601))
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

        # generate the input file for RU_hot thermal analysis
        generator = InputFileGenerator()
        generator.set_template_file('RU_hot_template.txt')
        generator.set_generated_file('RU_hot.d')
        generator.mark_anchor("$LOCALS")
        generator.transfer_var(length, 21, 3)
        generator.transfer_var(P_un, 25, 3)
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
        generator.transfer_var(GL102_601, 17, 3)
        generator.transfer_var(GL103_400, 45, 3)
        generator.generate()

        # the parent compute function actually runs the external code
        super(RU_hot, self).compute(inputs, outputs)

        # parse the output file from the external code and set the value of T_max
        parser = FileParser()
        parser.set_file('RU_hot.out')
        parser.mark_anchor("+RU_HOT")
        tBat = parser.transfer_var(7, 3)
        tMain = parser.transfer_var(34, 3)
        tProp = parser.transfer_var(35, 3)
        tTether = parser.transfer_var(36, 3)
        tBPanel = parser.transfer_var(41, 3)
        tDPanel = parser.transfer_var(43, 3)

        outputs['tBat'] = tBat
        outputs['tMain'] = tMain
        outputs['tProp'] = tProp
        outputs['tTether'] = tTether
        outputs['tBPanel'] = tBPanel
        outputs['tDPanel'] = tDPanel
if __name__ == "__main__":
    prob = om.Problem()
    model = prob.model

    # create and connect inputs and outputs
    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('length', val=0.2)
    indeps.add_output('P_un', val=1.0)
    indeps.add_output('eps', val=0.1)
    indeps.add_output('alp', val=0.4)
    indeps.add_output('GlMain', val=0.004)
    indeps.add_output('GlProp', val=0.004)
    indeps.add_output('GlTether', val=0.004)
    indeps.add_output('GlPanel', val=0.004)
    indeps.add_output('ci1', val=1.0)
    indeps.add_output('ci2', val=1.0)
    indeps.add_output('ci3', val=1.0)
    indeps.add_output('ci4', val=1.0)
    indeps.add_output('ci5', val=1.0)
    indeps.add_output('ci6', val=1.0)
    indeps.add_output('ci7', val=1.0)
    indeps.add_output('ci8', val=1.0)
    indeps.add_output('ci9', val=1.0)
    indeps.add_output('ci10', val=1.0)
    indeps.add_output('ci11', val=1.0)
    indeps.add_output('ci12', val=1.0) 


    model.add_subsystem('RU_hot', RU_hot(), promotes_inputs=['*'], promotes_outputs=[('tBat','tBat_h'), 
                        ('tMain','tMain_h'), ('tProp','tProp_h'), ('tBPanel', 'tBPanel_h'), ('tDPanel', 'tDPanel_h')])

    #run the ExternalCode Component once and print outputs
    prob.setup(check=True)
    prob.run_model()

    tBat =  prob['tBat_h']
    tProp =  prob['tProp_h']
    tMain =  prob['tMain_h']
    tBPanel = prob['tBPanel_h']
    tDPanel = prob['tDPanel_h']    
    print("Temperatures:, tBat={}, tProp={}, tMain={}, tBPanel={}, tDPanel={}".format(tBat, tProp, tMain, tBPanel, tDPanel))