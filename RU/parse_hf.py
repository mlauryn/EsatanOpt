"""
Parses the Esatan conductor report file (produced by REPORT_CONDUCTORS procedure) to prepare user-defined conductor data for openmdao 
"""
import re
import pandas as pd
from collections import defaultdict


# set up regular expressions
# use https://regexper.com to visualise these if required

def parse_hf(filepath):
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
    var_name = 'phi'
    
    rx_dict = {
    'var': re.compile(r'HF name = phi_(?P<var>\d+)'),
    'node': re.compile(r'Node (?P<node>\d+)')
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
    with open(filepath, 'r') as f:
        
        for line in f:  # This keeps reading the file line by line
            
            # at each line check for a match with a regex
            key, match = _parse_line(line)

            # extract training variable value
            if key == 'var':
                x = match.group('var')

            # extract node number
            if key == 'node':
                node = match.group('node')

                next(f)
                next(f)
                line = f.readline()

                while line.strip():

                    angle = line.split()[0]
                    #time = line.split()[1]
                    IS = line.split()[2]

                    entry = {
                    'var': float(x),
                    'node': int(node),
                    'angle': float(angle),
                    #'time': float(time),
                    'IS': float(IS)
                    }
                    # append the dictionary to the data list
                    data.append(entry)

                    line = f.readline() # This keeps reading the file
    
    """ output = defaultdict(dict)

    for entry in data:
        output[entry['angle']][entry['var']][entry['node']] = entry['IS'] """

    # create a pandas DataFrame from the list of dicts
    data = pd.DataFrame(data)
    # set the index
    data.set_index(['angle'], inplace=True)
    
    # consolidate df to remove nans
    #data = data.groupby(level=data.index.names).first()
    # upgrade Score from float to integer
    #data = data.apply(pd.to_numeric, errors='ignore')
    
    return data

if __name__ == '__main__':
    filepath = 'radiative_results.txt'
    df = parse_hf(filepath)

    #print(df)
    df1 = df.loc[0].pivot(index='var', columns='node', values='IS')  
    """ df_pivot = df1.pivot(index='var', columns='node', values='IS')
    df_pivot = df_pivot.filter(items=[1,2,3,4,5,46,65])
    a = df_pivot.to_numpy()
    idx = df_pivot.index.to_numpy() """

    print(df1)