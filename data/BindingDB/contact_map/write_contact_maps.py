import numpy as np
import pandas as pd

from urllib.request import urlopen
from progressbar import progressbar
from Bio.PDB import *


parser = PDBParser(PERMISSIVE=1)

def get_pdb_file(sequence):
    pdb_id = sequence_mapping[sequence].split(",")[0]
    try:
        return parser.get_structure(pdb_id,
                                    f"pdb_files/pdb{pdb_id.lower()}.ent")
    except:
        raise Exception(f"{pdb_id} has no file.")


def get_coord_list(residue):
    pass


def get_distance_matrix(coord_list):
    pass


def save_contact_map(sequence):
    pdb_file  = get_pdb_file(sequence)
    """
    coord_list = get_coord_list(pdb_file)
    return get_distance_matrix(coord_list)
    """


# Get BindingDB sequences and PDB IDs.
bindingdb_sequences = pd.read_pickle("../data_pre/mapped_bindingdb_sequences.pkl")
bindingdb_sequences = [element.split("\n")[2] for element in bindingdb_sequences]

sequence_mapping = pd.read_pickle("sequence_to_id_map.pkl")


for sequence in progressbar(bindingdb_sequences[:1]):
    save_contact_map(sequence)
