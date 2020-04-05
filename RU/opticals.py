
def opticals(node_grp, keys, optprop):

    
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


if __name__ == '__main__':
    
    from inits import nodes
    from ViewFactors import parse_vf
    
    filepath = 'ViewFactors.txt'
    optprop = parse_vf(filepath)

    nn, node_grp = nodes()
    
    #node_grp.update({'my_group': [54,55,56,57]})
    keys = ['Box:outer', 'Panel_outer:back']
    a = opticals(node_grp, keys, optprop)
    print(a)
    



