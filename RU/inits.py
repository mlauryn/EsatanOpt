"""
Generates initial conditions from Esatan model for openmdao analysis 
"""
import re
import pandas as pd
import numpy as np

def nodes(data, env='99999', inact='99998' ):
    """
    Reads Esatan node data csv export file and returns number of nodes in the model and node numbers grouped by node labels

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
        dictionary of node labels and node number list

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

def inits(data, env='99999', inact='99998'):
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

def conductors(nn, data, env='99999', inact='99998'):
    """
    Parse Esatan conductor data csv export file at given filepath

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


if __name__ == '__main__':
    
    node_data = 'nodal_data.csv'
    cond_data = 'Cond_data.csv'
    nn, groups = nodes(node_data)
    GL_init, GR_init = conductors(nn, cond_data) 
    QI_init, QS_init = inits(node_data)
    #print(nn)
    #print(nodes)
    for name in groups:
        print(name)