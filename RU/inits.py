"""
Generates initial conditions from Esatan model for openmdao analysis 
"""
import re
import pandas as pd
import numpy as np

def inits(n, nodes, conductors, env='99999', inact='99998'):
    """
    Parse Esatan conductor report file at given filepath

    Parameters
    ----------
    n   :   int
        number of nodes in the model (excluding deep space)
    env :   str
        environment (deep space) node number in Esatan model
    inact :   str
        inactive node number in Esatan model
    nodes : str
        Filepath for the Esatan nodal output csv file (must include at least QI, QS entities)
    conductors  : str
        Filepath for the Esatan conductor output csv file (must include at least GR, GL entities)

    Returns
    -------
    GL_init : n+1 x n+1 numpy array
        Linear conductor matrix (symmetric)
    GR_init : n+1 x n+1 numpy array
        Radiative exchange factor (REF) matrix (non-symmetric)
    QI_init :   n x 1 numpy array
        Internal heat source vector
    QS_init :   n x 1 numpy array
        Solar flux heat source vector

    """

    # prepare linear conductors
    
    df = pd.read_csv(conductors, header=1)

    GLs = df.filter(regex ='GL')
    GLs.columns = GLs.columns.str.extract(r'(\d+;\d+)', expand=False) # leave only indices in the column names

    GL_init = np.zeros((n+1,n+1))

    for Name, Data in GLs.iteritems():
        i = int(Name.split(';')[0])
        j = int(Name.split(';')[1])
        idx = (i,j)
        GL_init[idx] = Data.values

    # prepare radiative conductors

    GRs = df.filter(regex ='GR')
    GRs.columns = GRs.columns.str.extract(r'(\d+;\d+)', expand=False) # leave only indices in the column names

    GRs.columns = GRs.columns.str.replace(env, '0') #replace default esatan env node number to 0

    GRs = GRs.drop(GRs.filter(regex=inact).columns, axis=1) # drop inactive nodes

    GR_init = np.zeros((n+1,n+1))

    for Name, Data in GRs.iteritems():
        i = int(Name.split(';')[0])
        j = int(Name.split(';')[1])
        idx = (i,j)
        GR_init[idx] = Data.values
    
    #esatan does not include stefan-boltzman const
    sigma = 5.670374e-8
    GR_init = GR_init*sigma 

    # prepare initial boundary conditions 
    df = pd.read_csv(nodes, header=1)

    QI = df.filter(regex = 'QI(?!{})(?!{})'.format(env, inact)) # get rid of environment and inactive nodes
    QI_init = QI.to_numpy()[0]
    QI_init = np.insert(QI_init, 0, 0) # insert zero for deep space node 

    QS = df.filter(regex = 'QS(?!{})(?!{})(?!I)'.format(env, inact))
    QS_init = QS.to_numpy()[0]
    QS_init = np.insert(QS_init, 0, 0)
    
    return GL_init, GR_init, QI_init, QS_init


if __name__ == '__main__':
    n = 13
    nodes = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'
    GL_init, GR_init, QI_init, QS_init = inits(n, nodes, conductors)
    print(GR_init)