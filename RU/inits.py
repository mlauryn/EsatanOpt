"""
Generates initial conditions from Esatan model for openmdao analysis 
"""
import re
import pandas as pd
import numpy as np

def inits(n, env, inact, nodes, conductors):
    """
    Parse Esatan conductor report file at given filepath

    Parameters
    ----------
    n   :   int
        number of nodes in the model
    env :   str
        environment (deep space) node number
    inact :   str
        inactive node number
    nodes : str
        Filepath for the Esatan nodal output csv file (must include at least QI, QS entities)
    conductors  : str
        Filepath for the Esatan conductor output csv file (must include at least GR, GL entities)

    Returns
    -------
    GL_init : n x n numpy array
        Linear conductor matrix (symmetric)
    GR_init : n x n numpy array
        Radiative exchange factor (REF) matrix (non-symmetric)
    QI_init :   n x 1 numpy array
        Internal heat source vector
    QS_init :   n x 1 numpy array
        Solar flux heat source vector

    """

    # prepare linear conductors
    df = pd.read_csv(conductors, header=1)

    GLs = df.filter(regex ='GL')
    GLs.columns = GLs.columns.str.extract(r'(\d+;\d+)', expand=False)

    GL_init = np.zeros((n+1,n+1))

    for Name, Data in GLs.iteritems():
        i = int(Name.split(';')[0])
        j = int(Name.split(';')[1])
        idx = (i,j)
        GL_init[idx] = Data.values

    # prepare radiative conductors

    GRs = df.filter(regex ='GR')
    GRs.columns = GRs.columns.str.extract(r'(\d+;\d+)', expand=False)

    #replace default esatan env node number to 0
    GRs.columns = GRs.columns.str.replace(env, '0')

    # drop inactive nodes
    GRs = GRs.drop(GRs.filter(regex=inact).columns, axis=1)

    GR_init = np.zeros((n+1,n+1))

    for Name, Data in GRs.iteritems():
        i = int(Name.split(';')[0])
        j = int(Name.split(';')[1])
        idx = (i,j)
        GR_init[idx] = Data.values

    # prepare initial boundary conditions 
    df = pd.read_csv(nodes, header=1)

    QI = df.filter(regex = 'QI(?!{})(?!{})'.format(env, inact))
    QI_init = QI.to_numpy()[0]

    QS = df.filter(regex = 'QS(?!{})(?!{})(?!I)'.format(env, inact))
    QS_init = QS.to_numpy()[0]
    
    return GL_init, GR_init, QI_init, QS_init


if __name__ == '__main__':
    n = 13
    env = '99999'
    inact = '99998'
    nodes = 'Nodal_data.csv'
    conductors = 'Cond_data.csv'
    GL_init, GR_init, QI_init, QS_init = inits(n, env, inact, nodes, conductors)
    print(QS_init)