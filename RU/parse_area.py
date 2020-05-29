import numpy as np
file_name='area.txt'
area = []   
nodes =[]
with open(file_name, 'r') as f:
    for i in range(11):
        f.readline()
    
    line = f.readline()
    while line.strip():
        row = line.split('|')
        n = row[2].strip()
        a = row[-1].strip()
        area.append(a)
        nodes.append(n)
        f.readline()
        line = f.readline()
area = list(map(float, area))
nodes = list(map(int, nodes))
area_n = np.array(nodes, dtype=int)
area_val = np.array(area)
data_raw = np.vstack((area_n, area_val))
data = np.unique(data_raw, axis=1)
print(data,np.shape(data))                

