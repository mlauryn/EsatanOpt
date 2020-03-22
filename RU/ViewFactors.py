"""
Parses the Esatan conductor report file (produced by REPORT_CONDUCTORS procedure) to prepare user-defined conductor data for openmdao 
"""
import re
import pandas as pd

# set up regular expressions
# use https://regexper.com to visualise these if required

def parse_vf(filepath):
    """
    Parse Esatan view factor report file at given filepath

    Parameters
    ----------
    filepath : str
        Filepath for view factorreport file to be parsed

    Returns
    -------
    data : pd.DataFrame
        Parsed data

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

    data = []  # create an empty list to collect the data
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
                vf = match.group('vf')
            
                # create a dictionary containing this row of data
                row = {
                    'node number': int(node),
                    'area': float(area),
                    'emissivity': float(eps),
                    'vf': float(vf)
                }
                
                # append the dictionary to the data list
                data.append(row)

        """ # create a pandas DataFrame from the list of dicts
        data = pd.DataFrame(data)
        # set the index
        data.set_index(['node'], inplace=True)
        # consolidate df to remove nans
        data = data.groupby(level=data.index.names).first()
        # upgrade Score from float to integer
        #data = data.apply(pd.to_numeric, errors='ignore') """
    return data

if __name__ == '__main__':
    filepath = 'ViewFactors.txt'
    data = parse_vf(filepath)
    nodes = []
    area = []
    vf = []
    eps = []

    for entry in data:
        nodes.append(entry['node number'])
        area.append(entry['area'])
        vf.append(entry['vf']) 
        eps.append(entry['emissivity'])  
    print(nodes, area, vf, eps)