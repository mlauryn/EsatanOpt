import re
import pandas as pd
import numpy as np
from collections import defaultdict

def nodes(data='nodes_RU_v4_detail_cc.csv', env='99999', inact='99998' ):
    """
    Reads Esatan node data csv export file and returns dict of node numbers grouped by node labels
    Note: labels need to be defined in Esatan model

    Parameters
    ----------
    
    env :   str
        environment (deep space) node number in Esatan model
    inact :   str
        inactive node number in Esatan model
    data : str
        Filepath for the Esatan nodal output csv file (must include node labels)

    Returns
    -------
    nn   :   int
        number of thermal nodes in the model excluding deep space and inactive nodes

    labels : dict
        dictionary of node numbers lists grouped by node labels

    """
    
    # read node data

    df = pd.read_csv(data, header=1)

    # extract node labels

    nodes = df.filter(regex = 'L(?!{})(?!{})(\d+)'.format(env, inact) )

    nodes.columns = nodes.columns.str.extract(r'L(\d+)', expand=False) # keep just node number

    node_dict = nodes.to_dict('records')[0]

    nn = len(node_dict) # number of nodes

    labels = {}

    # group nodes by labels
    for key, value in sorted(node_dict.items()):
        labels.setdefault(value, []).append(key)

    # change node numbers from strings to int
    for key in labels:
        labels[key] = list(map(int, (labels[key])))
    
    return nn, labels

def inits(data='nodal_detail_cc.csv', env='99999', inact='99998'):
    """
    Reads Esatan node data csv export file and returns initial heat load boundary conditions

    Parameters
    ----------
    
    env :   str
        environment (deep space) node number in Esatan model
    inact :   str
        inactive node number in Esatan model
    data : str
        Filepath for the Esatan nodal output csv file (must include at least QI, QS entities)

    Returns
    -------

    QI_init :   n x 1 numpy array
        Internal heat source vector
    QS_init :   n x 1 numpy array
        Solar flux heat source vector

    """

    # read node data

    df = pd.read_csv(data, header=1)

    # prepare initial boundary conditions 

    QI = df.filter(regex = 'QI(?!{})(?!{})'.format(env, inact)) # get rid of environment and inactive nodes
    QI_init = QI.to_numpy()[0]

    QI_init = np.insert(QI_init, 0, 0) # insert zero for deep space node
    QI_init = QI_init[np.newaxis,:].T # make array 2D


    QS = df.filter(regex = 'QS(?!{})(?!{})(?!I)'.format(env, inact))
    QS_init = QS.to_numpy()[0]
    QS_init = np.insert(QS_init, 0, 0)
    QS_init = QS_init[np.newaxis,:].T

    return QI_init, QS_init

def conductors(nn, data='Cond_detail_cc.csv', env='99999', inact='99998'):
    """
    Parse Esatan conductor data csv export file at given filepath and returns GL and GR conductors
    Note: At least GL and GR entities need to be selected in output calls, print conductor CSV options

    Parameters
    ----------
    
    env :   str
        environment (deep space) node number in Esatan model
    inact :   str
        inactive node number in Esatan model
    data  : str
        Filepath for the Esatan conductor output csv file (must include at least GR, GL entities)
    nn   :   int
        number of thermal nodes in the model excluding deep space and inactive nodes

    Returns
    -------
    GL_init : n+1 x n+1 numpy array
        Linear conductor matrix (symmetric)
    GR_init : n+1 x n+1 numpy array
        Radiative exchange factor (REF) matrix (non-symmetric)

    """

    # prepare linear conductors
    
    df = pd.read_csv(data, header=1)

    GLs = df.filter(regex ='GL')
    GLs.columns = GLs.columns.str.extract(r'(\d+;\d+)', expand=False) # leave only indices in the column names

    GL_init = np.zeros((nn+1,nn+1))

    for Name, Data in GLs.iteritems():
        i = int(Name.split(';')[0])
        j = int(Name.split(';')[1])
        idx = (i,j)
        GL_init[idx] = Data.values
    
    #define GL diagonal elements as negative sum of respective row (all node conductor couplings)
    diag = np.negative(np.sum(GL_init, 1))
    di = np.diag_indices(nn+1)
    GL_init[di] = diag

    GL_init[0,0] = 1.0 # deep space node temperature = 0 K (this coef is needed to avoid singularity in heat equations)

    # prepare radiative conductors

    GRs = df.filter(regex ='GR')
    GRs.columns = GRs.columns.str.extract(r'(\d+;\d+)', expand=False) # leave only indices in the column names

    GRs.columns = GRs.columns.str.replace(env, '0') #replace default esatan deep space node number to 0

    GRs = GRs.drop(GRs.filter(regex=inact).columns, axis=1) # drop inactive nodes

    GR_init = np.zeros((nn+1,nn+1))

    for Name, Data in GRs.iteritems():
        i = int(Name.split(';')[0])
        j = int(Name.split(';')[1])
        idx = (i,j)
        GR_init[idx] = Data.values
    
    #esatan does not include stefan-boltzman const in GR output
    sigma = 5.670374e-8
    GR_init = GR_init*sigma 

    GR_init[0,:] = 0. # REFs from deep space = 0
    GR_init = GR_init.T # needs to be transposed to result in proper equilibrium equations

    # REF entries on the main diagonal must be formed by the negative sum of the respective column 
    diag = np.negative(np.sum(GR_init, 0))
    di = np.diag_indices(nn+1) 
    GR_init[di] = diag
    
    return GL_init, GR_init


# set up regular expressions
# use https://regexper.com to visualise these if required

def parse_vf(filepath='ViewFactors.txt'):
    """
    This script parses the Esatan view factor report file (produced by REPORT_VF procedure) to collect data required to calculate radiative exchange factors to deep space.
    Note:
    only surfaces exposed to deep space are collected (if VF to environment = 0, the node is droped), thus unwanted surfaces have to be either conductive or inactive
    For proper results each emitting face must have unique node number (different nodes for surface 1 and 2 if both sides are emitting)
    "Report against thermal nodes" option in the report menu has to bee unticked

    Parameters
    ----------
    filepath : str
        Filepath for view factor report file to be parsed

    Returns
    -------
    data : list of dict of node areas, emissivities and view factors grouped by node numbers as keys

    """
    rx_dict = {
    'node number': re.compile(r'Thermal Node = (?P<node>\d*)\n'),
    'area': re.compile(r'Area = (?P<area>\d.\d*)'),
    'emissivity': re.compile(r'Emissivity = (?P<eps>\d.\d*)'),
    'view factor': re.compile(r'VF to environment = (?P<vf>\d.\d*)')
    }

    def _parse_line(line):
        """
        Do a regex search against all defined regexes and
        return the key and match result of the first matching regex

        """

        for key, rx in rx_dict.items():
            match = rx.search(line)
            if match:
                return key, match
        # if there are no matches
        return None, None

    data = {}  # create an empty dict to collect the data
    # open the file and read through it line by line
    with open(filepath, 'r') as file_object:
        
        for line in file_object:
            
            # at each line check for a match with a regex
            key, match = _parse_line(line)

            # extract node number
            if key == 'node number':
                node = match.group('node')

            # extract area and emissivity
            if key == 'area':
                area = match.group('area')

                key, match = _parse_line(line.split(',')[1])
                if key == 'emissivity':
                    eps = match.group('eps')
                            
            #extract view factor    
            if key == 'view factor':
                vf = float(match.group('vf'))

                if vf == 0.0:
                    continue # filter out internal surfacess
            
                # create a dictionary containing this row of data
                entry = {int(node): {'area': float(area), 'eps': float(eps), 'vf': vf}}
                
                # append the dictionary to the data list
                data.update(entry)

    return data

def opticals(node_grp, keys, optprop):

    # new dict of user selected node labels
    node_groups = {new_key: node_grp[new_key] for new_key in keys}

    faces = []

    for grp in node_groups:
        
        areas =[]
        VFs = []
        emissivities = []

        for node in node_groups[grp]:
            area = optprop[node]['area']
            vf = optprop[node]['vf']
            eps = optprop[node]['eps']
            areas.append(area)
            VFs.append(vf)
            emissivities.append(eps)

        entry = {'name':grp, 'nodes':node_groups[grp], 'areas':areas, 'VFs': VFs, 'eps':emissivities}
        faces.append(entry)

    return faces

def parse_hf(filepath='Radiative_results.txt', end=90.0, num=19):
    """
    Parse Esatan radiative restults report file at given filepath

    Parameters
    ----------
    filepath : str
        Filepath for report file to be parsed

    Returns
    -------
    train_data : numpy array
        Parsed training data for surrogate model

    """

    xt = np.linspace(0., end, num) # training variables

    rx = re.compile(r'Node (?P<node>\d+)')

    data = []  # create an empty list to collect the data
    # open the file and read through it line by line
    with open(filepath, 'r') as f:
        node = ''
        for line in f:  # This keeps reading the file line by line
            
            # at each line check for a match with a regex of a node number
            match = rx.search(line)
            if match:
                if match.group('node') == node: # check for internal faces and drop
                    continue
                else:
                    node = match.group('node')
                next(f)
                next(f)
                line = f.readline()

                while line.strip(): # This keeps reading until end of table

                    data.append(line.split()[2])

                    line = f.readline()
        
        q_s = list(map(float, data))

        yt = np.array(q_s).reshape((int(node),num))

        train_data = np.vstack([xt,yt]).T
    
    return train_data

def parse_cond(filepath='conductors.txt'):
    """
    Parses the Esatan conductor report file (produced by REPORT_CONDUCTORS procedure) to prepare user-defined conductor data for openmdao 

    Parameters
    ----------
    filepath : str
        Filepath for report file to be parsed

    Returns
    -------
    data : list
        Parsed data

    """
    # set up regular expressions
    # use https://regexper.com to visualise these if required

    rx_dict = {
    'cond_name': re.compile(r'USER DEFINED CONDUCTOR = (?P<name>.*)\n'),
    'cond_type': re.compile(r'TYPE\s*= (?P<cond_type>.*)\n'),
    'nodes': re.compile(r'\[(?P<nodes>\d.*)\]'),
    'shape factor': re.compile(r'Shape Factor: (?P<sf>\d.\d*)'),
    'conductivity': re.compile(r'Conductivity: (?P<cond>\d.\d*)'),
    }

    def _parse_line(line):
        """
        Do a regex search against all defined regexes and
        return the key and match result of the first matching regex

        """

        for key, rx in rx_dict.items():
            match = rx.search(line)
            if match:
                return key, match
        # if there are no matches
        return None, None

    data = []  # create an empty list to collect the data
    # open the file and read through it line by line
    with open(filepath, 'r') as file_object:
        
        # Skips text before the beginning of the user links block:
        
        for line in file_object:
            if line.strip() == 'USER-DEFINED LINKS':  
                break
        
        for line in file_object:  # This keeps reading the file
            
            # at each line check for a match with a regex
            key, match = _parse_line(line)

            # extract conductor name
            if key == 'cond_name':
                cond = match.group('name')

            # extract cond_type
            if key == 'cond_type':
                cond_type = match.group('cond_type')
            
            # extract nodes
            if key == 'nodes':
                
                nodes = []
                shape_factors = []
                values = []

                # read each line of the table until a blank line
                while line.strip():

                    key, match = _parse_line(line.split('=')[0])

                    node_pair = tuple(map(int, match.group('nodes').split(',')))

                    key, match = _parse_line(line.split('=')[1])
                                
                    #extract shape factor    
                                    
                    SF = float(match.group('sf'))
                
                    if SF == 0.0:
                        SF = 1.0 # override by user

                        k = line.split('=')[1].split(',')[2].split(':')[1].strip() # conductivity equals user overriden value

                    else:
                        key, match = _parse_line(line.split('=')[1].split(',')[1])
                        k = match.group('cond') # else use normal conductivity
                    
                    nodes.append(node_pair)
                    shape_factors.append(SF)
                    values.append(float(k))

                    line = file_object.readline()

                # create a dictionary containing this conductor data
                entry = {
                    'cond_name': cond,
                    'cond_type': cond_type,
                    'nodes': nodes,
                    'SF': shape_factors,
                    'values': values
                }
                # append the dictionary to the data list
                data.append(entry)

                line = file_object.readline() # This keeps reading the file
            
            # keeps reading until 'contact zone' block

            if line.strip() == 'CONTACT-ZONE LINKS':
                break

    return data

def idx_dict(src, ref):
    """
    Group node value indexes in src by corresponding node labels in ref

    """
    indices = defaultdict(list)

    for idx, val in enumerate(src):
        for name in ref:
            if val not in ref[name]:
                continue

            indices[name].append(idx)

    return indices