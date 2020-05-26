""" This component takes thermal conductivities k as input and generates linear conductors GL = k * SF, 
where SF is shape factor given by SF = A/L (A-crossectional area of conductor, L - length of conductor) """

import openmdao.api as om
import numpy as np

class LinearCondComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('SF', types=list, desc='list of shape factors for output conductors')
    
    def setup(self):
        n_in = len(self.options['SF'])
        self.add_output('GLs', shape=n_in)
        self.add_input('k', shape=n_in)
        rows = np.arange(n_in)
        cols = rows
        self.declare_partials('GLs', 'k', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        SF = self.options['SF']
        outputs['GLs'] = SF * inputs['k']

    def compute_partials(self, inputs, partials):
        SF = self.options['SF']
        partials['GLs', 'k'] = SF

#for testing
if __name__ == "__main__":
    model = om.Group()
    n = 4
    SF = list(range(1,n))
    comp = om.IndepVarComp()
    comp.add_output('k', shape=n)

    model.add_subsystem('input', comp)
    model.add_subsystem('example', LinearCondComp(SF=SF))

    model.connect('input.k', 'example.k')
    
    problem = om.Problem(model=model)
    problem.setup()
    problem.run_model()
    totals = problem.compute_totals(['example.GLs'], ['input.k'])

    print(totals['example.GLs', 'input.k'])


