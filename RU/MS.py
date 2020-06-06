import os
import openmdao.api as om
import numpy as np
from Pre_process import parse_vf, parse_cond, inits, conductors, nodes, opticals, idx_dict, parse_ar
from Solar import Solar
from Cond_group import Cond_group
#from Thermal_Cycle import Thermal_Cycle
from Thermal_direct import Thermal_direct
from PowerOutput import PowerOutput
from PowerInput import PowerInput

class MainSP(om.Group):
    """ 
    Main s/c thermal model with coupling
    """
    def __init__(self, npts, model, labels):
        super(MainSP, self).__init__()
        
        fpath = os.path.dirname(os.path.realpath(__file__))
        model_dir = fpath + '/Esatan_models/' + model

        self.nn, self.groups, outp = nodes(data=model_dir+'/nodes_output.csv')
        self.npts = npts
        self.model = model

        # user defined conductors
        self.user_cond = parse_cond(filepath=model_dir+'/cond_report.txt')
        
        # initial conductor values
        self.GL_init, self.GR_init = conductors(nn=self.nn, data=model_dir+'/cond_output.csv')

        #optical properties
        optprop = parse_vf(filepath=model_dir+'/vf_report.txt')
        areas = parse_ar(filepath=model_dir+'/area.txt')
        self.faces = opticals(self.groups, labels, optprop, areas) 
        surf_nodes = []
        surf_area = []
        for face in self.faces:
            surf_nodes.extend(face['nodes'])
            surf_area.extend(face['areas'])

        self.surf_nodes = np.array(surf_nodes)
        surf_area = np.array(surf_area)
        self.surf_area = surf_area[self.surf_nodes.argsort()] # sort areas by ascending node number
        self.surf_nodes.sort() # sort node numbers ascending

    def setup(self):
        nn = self.nn
        npts = self.npts
        n_in = len(self.surf_nodes)
        groups = self.groups

        # input variables
        params = self.add_subsystem('params', om.IndepVarComp(), promotes=['*'])
        params.add_output('QI', val=np.zeros((nn+1, npts)), units='W')
        params.add_output('phi', val=np.zeros(npts), units='deg' )
        params.add_output('dist', val=np.ones(npts))
        params.add_output('alp_r', val=np.zeros(n_in), desc='absorbtivity of the input node radiating surface')
        params.add_output('cr', val=np.zeros(n_in), desc='solar cell or radiator installation decision for input nodes')
        for cond in self.user_cond:
            params.add_output(cond['cond_name'], val=cond['values'][0] ) # adds output variable with the same name as user conductor name
        for face in self.faces:
            params.add_output(face['name'], val=face['eps'][0] ) # adds independant variable as face name and assigns emissivity of it's first node
        params.add_output('eta', val=np.ones((n_in,npts))*0.275/0.91, desc='solar cell efficiency with respect to absorbed power for input surface nodes over time ')

        self.add_subsystem('Cond', Cond_group(n=nn, conductors=self.user_cond, faces=self.faces, 
                            GL_init=self.GL_init, GR_init=self.GR_init), promotes=['*'])
        self.add_subsystem('Solar', Solar(npts=npts, areas=self.surf_area, nodes=self.surf_nodes, model=self.model), promotes=['*'])
        #self.add_subsystem('Thermal', Thermal_Cycle(nn=nn, npts=npts, nodes=self.surf_nodes), promotes=['*'])
        self.add_subsystem('Thermal', Thermal_direct(nn=nn, npts=npts, nodes=self.surf_nodes), promotes=['*'])
        self.add_subsystem('Pout', PowerOutput(nn=nn, npts=npts), promotes=['*'])
        self.add_subsystem('Pin', PowerInput(n_in=n_in, npts=npts), promotes=['*'])

        # equality contraint for keeping Pin=Pout (conservation of energy)
        equal = om.EQConstraintComp()
        self.add_subsystem('equal', equal)
        equal.add_eq_output('power_bal', shape=npts, add_constraint=True, normalize=False, eq_units='W')
        self.connect('P_out', 'equal.lhs:power_bal')
        self.connect('P_in', 'equal.rhs:power_bal')

        # global indices for components
        obc_nodes = groups['PCB']
        prop_nodes = groups['Propulsion_bot']
        bat_nodes = groups['Battery']
        ins_nodes = groups['Instrument_inner']
        es_nodes = groups['Esail_bot']
        trx_nodes = groups['TRx']
        aocs_nodes = groups['AOCS'] + groups['RW_Z']

        flat_indices = np.arange(0,(nn+1)*npts).reshape((nn+1,npts))
        obc_idx = flat_indices[obc_nodes,:]
        prop_idx = flat_indices[prop_nodes,:]
        bat_idx = flat_indices[bat_nodes,:]
        ins_idx = flat_indices[ins_nodes,:]
        es_idx = flat_indices[es_nodes,:]
        trx_idx = flat_indices[trx_nodes,-1]
        aocs_idx = flat_indices[aocs_nodes,:]


        # objective function
        self.add_subsystem('TRx_power', om.ExecComp('P_trx = -sum(QI_trx)', QI_trx=np.ones(len(trx_nodes))), promotes=['*'])
        self.connect('QI', 'QI_trx', src_indices=trx_idx, flat_src_indices=True)

        # temperature constraint aggregation Kreisselmeier-Steinhauser Function
        self.add_subsystem('bat_lwr', om.KSComp(width=npts, vec_size=len(bat_nodes), upper=273., lower_flag=True))
        self.add_subsystem('bat_upr', om.KSComp(width=npts, vec_size=len(bat_nodes), upper=45.+273.))
        self.add_subsystem('prop_upr', om.KSComp(width=npts, vec_size=len(prop_nodes), upper=80.+273.))
        self.add_subsystem('prop_lwr', om.KSComp(width=npts, vec_size=len(prop_nodes), upper=-10.+273., lower_flag=True))
        self.add_subsystem('obc_upr', om.KSComp(width=npts, vec_size=len(obc_nodes), upper=85+273.))
        self.add_subsystem('obc_lwr', om.KSComp(width=npts, vec_size=len(obc_nodes), upper=-40.+273., lower_flag=True))
        self.add_subsystem('ins_upr', om.KSComp(width=npts, vec_size=len(ins_nodes), upper=60.+273.))
        self.add_subsystem('ins_lwr', om.KSComp(width=npts, vec_size=len(ins_nodes), upper=-20.+273., lower_flag=True))
        self.add_subsystem('es_upr', om.KSComp(width=npts, vec_size=len(es_nodes), upper=85.+273.))
        self.add_subsystem('es_lwr', om.KSComp(width=npts, vec_size=len(es_nodes), upper=-40.+273., lower_flag=True))
        self.add_subsystem('trx_upr', om.KSComp(width=npts, vec_size=len(trx_nodes), upper=50.+273.))
        self.add_subsystem('trx_lwr', om.KSComp(width=npts, vec_size=len(trx_nodes), upper=-20.+273., lower_flag=True))
        self.add_subsystem('aocs_upr', om.KSComp(width=npts, vec_size=len(aocs_nodes), upper=70.+273.))
        self.add_subsystem('aocs_lwr', om.KSComp(width=npts, vec_size=len(aocs_nodes), upper=-40.+273., lower_flag=True))

        # subsystem power constraint
        self.add_subsystem('obc_pwr', om.KSComp(width=npts, vec_size=len(obc_nodes), upper=0.5/len(obc_nodes), lower_flag=True))
        self.add_subsystem('aocs_pwr', om.KSComp(width=npts, vec_size=len(aocs_nodes), upper=1.0/len(aocs_nodes), lower_flag=True))
        self.add_subsystem('prop_pwr', om.KSComp(width=npts, vec_size=len(prop_nodes), upper=0.5/len(prop_nodes), lower_flag=True))
        self.add_subsystem('ins_pwr', om.KSComp(width=npts, vec_size=len(ins_nodes), upper=1.0/len(ins_nodes), lower_flag=True))

        # KS connections
        self.connect('T', 'bat_lwr.g', src_indices=bat_idx, flat_src_indices=True)
        self.connect('T', 'bat_upr.g', src_indices=bat_idx, flat_src_indices=True)
        self.connect('T', 'prop_lwr.g', src_indices=prop_idx, flat_src_indices=True)
        self.connect('T', 'prop_upr.g', src_indices=prop_idx, flat_src_indices=True)
        self.connect('T', 'obc_lwr.g', src_indices=obc_idx, flat_src_indices=True)
        self.connect('T', 'obc_upr.g', src_indices=obc_idx, flat_src_indices=True)
        self.connect('T', 'ins_lwr.g', src_indices=ins_idx, flat_src_indices=True)
        self.connect('T', 'ins_upr.g', src_indices=ins_idx, flat_src_indices=True)
        self.connect('T', 'es_lwr.g', src_indices=es_idx, flat_src_indices=True)
        self.connect('T', 'es_upr.g', src_indices=es_idx, flat_src_indices=True)
        self.connect('T', 'trx_lwr.g', src_indices=trx_idx, flat_src_indices=True)
        self.connect('T', 'trx_upr.g', src_indices=trx_idx, flat_src_indices=True)
        self.connect('T', 'aocs_lwr.g', src_indices=aocs_idx, flat_src_indices=True)
        self.connect('T', 'aocs_upr.g', src_indices=aocs_idx, flat_src_indices=True)

        self.connect('QI', 'obc_pwr.g', src_indices=obc_idx, flat_src_indices=True)
        self.connect('QI', 'aocs_pwr.g', src_indices=aocs_idx, flat_src_indices=True)
        self.connect('QI', 'prop_pwr.g', src_indices=prop_idx, flat_src_indices=True)
        self.connect('QI', 'ins_pwr.g', src_indices=ins_idx, flat_src_indices=True)
