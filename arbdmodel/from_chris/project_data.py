import numpy as np

""" Info for making the pdb model """
pdb_list = ['4f5s']
pdb_copy_number = {'4f5s':1}
pdb_copy_number_total = sum([v for k,v in pdb_copy_number.items()])

def get_copy_number( pdb_key, total_proteins ):
    if pdb_key not in pdb_copy_number:
        raise ValueError
    return int(np.round( pdb_copy_number[pdb_key] *
                         total_proteins/ pdb_copy_number_total ))
