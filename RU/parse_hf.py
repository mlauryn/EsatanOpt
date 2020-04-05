import re
import numpy as np

def parse_hf(filepath, end=90.0, num=19):
    """
    Parse Esatan radiative restults report file at given filepath

    Parameters
    ----------
    filepath : str
        Filepath for report file to be parsed

    Returns
    -------
    data : pd.DataFrame
        Parsed data

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

if __name__ == '__main__':
    filepath = 'radiative_results.txt'
    y = parse_hf(filepath)

    print(y)