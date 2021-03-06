"""
Parses the Esatan conductor report file (produced by REPORT_CONDUCTORS procedure) to prepare user-defined conductor data for openmdao 
"""
import re
import pandas as pd

# set up regular expressions
# use https://regexper.com to visualise these if required

def parse_cond(filepath):
    """
    Parse Esatan conductor report file at given filepath

    Parameters
    ----------
    filepath : str
        Filepath for report file to be parsed

    Returns
    -------
    data : pd.DataFrame
        Parsed data

    """
    rx_dict = {
    'cond_name': re.compile(r'USER DEFINED CONDUCTOR = (?P<name>.*)\n'),
    'type': re.compile(r'TYPE\s*= (?P<type>.*)\n'),
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
        # Reads text until the contact zone block:
        
        for line in file_object:  # This keeps reading the file
            
            # at each line check for a match with a regex
            key, match = _parse_line(line)

            # extract conductor name
            if key == 'cond_name':
                cond = match.group('name')

            # extract type
            if key == 'type':
                type = match.group('type')
            
            # extract nodes
            if key == 'nodes':
                nodes = tuple(map(int, match.group('nodes').split(',')))

                key, match = _parse_line(line.split('=')[1])
                            
                #extract shape factor    
                                 
                SF = float(match.group('sf'))
            
                if SF == 0.0:
                    SF = 1.0 # override by user

                    k = line.split('=')[1].split(',')[2].split(':')[1].strip() # conductivity equals user overriden value

                else:
                    key, match = _parse_line(line.split('=')[1].split(',')[1])
                    k = match.group('cond') # else use normal conductivity
                
                # create a dictionary containing this row of data
                row = {
                    'cond_name': cond,
                    'type': type,
                    'nodes': nodes,
                    'SF': SF,
                    'conductivity': float(k)
                }
                # append the dictionary to the data list
                data.append(row)

            if line.strip() == 'CONTACT-ZONE LINKS':
                break


        """ # create a pandas DataFrame from the list of dicts
        data = pd.DataFrame(data)
        # set the index
        data.set_index(['cond_name'], inplace=True)
        # consolidate df to remove nans
        data = data.groupby(level=data.index.names).first()
        # upgrade Score from float to integer
        #data = data.apply(pd.to_numeric, errors='ignore') """
    return data

if __name__ == '__main__':
    filepath = 'conductors.txt'
    data = parse_cond(filepath)
    nodes = {}
    shape_factors = {}
    values = {}
    for entry in data:
        nodes.update( {entry['cond_name'] : entry['nodes']} )
        shape_factors.update( {entry['cond_name'] : entry['SF'] } )
        values.update( {entry['cond_name'] : entry['conductivity'] } ) 
    print(nodes, shape_factors, values)