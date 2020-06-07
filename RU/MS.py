import os
import openmdao.api as om
import numpy as np
from Pre_process import parse_vf, parse_cond, inits, conductors, nodes, opticals, idx_dict, parse_ar
from Solar import Solar
from Cond_group import Cond_group
from Thermal_Cycle import Thermal_Cycle
#from Thermal_direct import Thermal_direct
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
        data = model_dir+'/nodes_output.csv'

        self.nn, self.groups, outp = nodes(data=data)
        self.npts = npts
        self.model = model

        # user defined conductors
        self.user_cond = parse_cond(filepath=model_dir+'/cond_report.txt')
        
        # initial conductor values
        self.GL_init, self.GR_init = conductors(nn=self.nn, data=model_dir+'/cond_output.csv')

        #initial boundary cond
        self.QI_init, self.QS_init = inits(data=data)

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
        self.n_in = len(self.surf_nodes)

        # index dictionary of radiative nodes_list
        idx = idx_dict(self.surf_nodes, self.groups)

        # local indices for external geometry nodes
        solar_cells = sum([idx[array] for array in [
            'SolarArray'
            ]], [])

        structure = sum([idx[geom] for geom in [
            'SP_Xplus_upr',
            'SP_Xminus_upr',
            'SP_Yplus_upr',
            'SP_Yminus_upr',
            'SP_Zplus_upr',
            'SP_Zminus_upr',
            ]], [])

        optics = sum([idx[geom] for geom in [
            'Telescope_outer',
            'StarTracker_outer',
            ]], [])

        inside = sum([idx[geom] for geom in [
            'SP_Yplus_lwr',
            'Instrument_outer',
            'BottomPlate_upr',
            ]], [])
        

        # initial values for some input variables
        self.cr_init = np.zeros(self.n_in)
        self.alp = np.zeros(self.n_in)

        self.cr_init[solar_cells] = 1.0
        self.alp[structure] = 0.1
        self.alp[inside] = 0.26
        self.alp[optics] = 0.88
        self.alp[idx['Propulsion_top']] = 0.72
        self.alp[idx['Esail_top']] = 1.

    def setup(self):
        nn = self.nn
        npts = self.npts
        n_in = self.n_in
        groups = self.groups

        # input variables
        params = self.add_subsystem('params', om.IndepVarComp(), promotes=['*'])
        params.add_output('QI', val=np.tile(self.QI_init, (1,npts)), units='W')
        params.add_output('phi', val=np.zeros(npts), units='deg' )
        params.add_output('dist', val=np.ones(npts))
        params.add_output('alp_r', val=self.alp, desc='absorbtivity of the input node radiating surface')
        params.add_output('cr', val=self.cr_init, desc='solar cell or radiator installation decision for input nodes')
        for cond in self.user_cond:
            params.add_output(cond['cond_name'], val=cond['values'][0] ) # adds output variable with the same name as user conductor name
        for face in self.faces:
            params.add_output(face['name'], val=face['eps'][0] ) # adds independant variable as face name and assigns emissivity of it's first node
        #params.add_output('eta', val=np.ones((n_in,npts))*0.275/0.91, desc='solar cell efficiency with respect to absorbed power for input surface nodes over time ')

        self.add_subsystem('Cond', Cond_group(n=nn, conductors=self.user_cond, faces=self.faces, 
                            GL_init=self.GL_init, GR_init=self.GR_init), promotes=['*'])
        self.add_subsystem('Solar', Solar(npts=npts, areas=self.surf_area, nodes=self.surf_nodes, model=self.model), promotes=['*'])
        self.add_subsystem('Thermal', Thermal_Cycle(nn=nn, npts=npts, nodes=self.surf_nodes), promotes=['*'])
        #self.add_subsystem('Thermal', Thermal_direct(nn=nn, npts=npts, nodes=self.surf_nodes), promotes=['*'])
        self.add_subsystem('Pout', PowerOutput(nn=nn, npts=npts), promotes=['*'])
        self.add_subsystem('Pin', PowerInput(n_in=n_in, npts=npts), promotes=['*'])

        # global indices for components
        obc_nodes = groups['PCB']
        prop_nodes = groups['Propulsion_bot']
        bat_nodes = groups['Battery']
        ins_nodes = groups['Instrument_inner']
        es_nodes = groups['Esail_bot']
        trx_nodes = groups['TRx']
        aocs_nodes = groups['AOCS'] #+ groups['RW_Z']
        equip_nodes = aocs_nodes + obc_nodes + ins_nodes + trx_nodes #+ es_nodes

        flat_indices = np.arange(0,(nn+1)*npts).reshape((nn+1,npts))
        obc_idx = flat_indices[obc_nodes,:]
        prop_idx = flat_indices[prop_nodes,:]
        bat_idx = flat_indices[bat_nodes,:]
        ins_idx = flat_indices[ins_nodes,:]
        es_idx = flat_indices[es_nodes,:]
        trx_idx = flat_indices[trx_nodes,:]
        aocs_idx = flat_indices[aocs_nodes,:]
        equip_idx = flat_indices[equip_nodes,:]

        # objective function
        self.add_subsystem('TRx_power', om.ExecComp('P_trx = -sum(QI_trx)', QI_trx=np.ones(len(trx_nodes))), promotes=['*'])
        self.connect('QI', 'QI_trx', src_indices=flat_indices[trx_nodes,-1], flat_src_indices=True)

        # temperature constraint aggregation Kreisselmeier-Steinhauser Function
        self.add_subsystem('bat_lwr', om.KSComp(width=npts, vec_size=len(bat_nodes), upper=273., lower_flag=True))
        self.add_subsystem('bat_upr', om.KSComp(width=npts, vec_size=len(bat_nodes), upper=45.+273.))
        self.add_subsystem('prop_upr', om.KSComp(width=npts, vec_size=len(prop_nodes), upper=80.+273.))
        self.add_subsystem('prop_lwr', om.KSComp(width=npts, vec_size=len(prop_nodes), upper=-10.+273., lower_flag=True))
        self.add_subsystem('equip_upr', om.KSComp(width=npts, vec_size=len(equip_nodes), upper=50+273.))
        self.add_subsystem('equip_lwr', om.KSComp(width=npts, vec_size=len(equip_nodes), upper=-20.+273., lower_flag=True))

        # subsystem power constraint
        self.add_subsystem('obc_pwr', om.KSComp(width=npts, vec_size=len(obc_nodes), upper=0.5/len(obc_nodes), lower_flag=True))
        self.add_subsystem('aocs_pwr', om.KSComp(width=npts, vec_size=len(aocs_nodes), upper=1.0/len(aocs_nodes), lower_flag=True))
        self.add_subsystem('prop_pwr', om.KSComp(width=npts, vec_size=len(prop_nodes), upper=0.5/len(prop_nodes), lower_flag=True))
        #self.add_subsystem('ins_pwr', om.KSComp(width=npts, vec_size=len(ins_nodes), upper=1.0/len(ins_nodes), lower_flag=True))
        #self.add_subsystem('es_pwr',  om.KSComp(width=1, vec_size=len(es_nodes), upper=3.5/len(es_nodes), lower_flag=True))

        # equality contraint for keeping Pin=Pout (conservation of energy)
        equal = om.EQConstraintComp()
        self.add_subsystem('equal', equal)
        equal.add_eq_output('power_bal', shape=npts, add_constraint=True, normalize=False, eq_units='W')
        self.connect('P_out', 'equal.lhs:power_bal')
        self.connect('P_in', 'equal.rhs:power_bal')

        # KS connections
        self.connect('T', 'bat_lwr.g', src_indices=bat_idx, flat_src_indices=True)
        self.connect('T', 'bat_upr.g', src_indices=bat_idx, flat_src_indices=True)
        self.connect('T', 'prop_lwr.g', src_indices=prop_idx, flat_src_indices=True)
        self.connect('T', 'prop_upr.g', src_indices=prop_idx, flat_src_indices=True)
        self.connect('T', 'equip_lwr.g', src_indices=equip_idx, flat_src_indices=True)
        self.connect('T', 'equip_upr.g', src_indices=equip_idx, flat_src_indices=True)

        self.connect('QI', 'obc_pwr.g', src_indices=obc_idx, flat_src_indices=True)
        self.connect('QI', 'aocs_pwr.g', src_indices=aocs_idx, flat_src_indices=True)
        self.connect('QI', 'prop_pwr.g', src_indices=prop_idx, flat_src_indices=True)
        #self.connect('QI', 'ins_pwr.g', src_indices=ins_idx, flat_src_indices=True)
        #self.connect('QI', 'es_pwr.g', src_indices=flat_indices[es_nodes,0], flat_src_indices=True) # valid only at Earth
