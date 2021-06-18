import numpy as np
import pandas as pd

from urllib.request import urlopen


def get_pdb_id(sequence):
    return sequence_mapping[sequence]


def get_pdb_file(pdb_id):
    response = urlopen(f"https://www.rcsb.org/fasta/entry/{target.upper()}/display")
    data = response.read().decode("utf-8").strip("\n")


def get_calpha_atom_coord(residue):
    pass


def get_distance_matrix(coord_list):
    pass


def get_contact_map(sequence):
    pdb_file  = get_pdb_file(get_pdb_id(sequence))
    coord_list = get_coord_list(pdb_file)
    return get_distance_matrix(coord_list)


# Get BindingDB sequences and PDB IDs.
bindingdb_sequences = pd.read_pickle("../data_pre/mapped_bindingdb_sequences.pkl")
bindingdb_sequences = [element.split("\n")[2] for element in bindingdb_sequences]

with open("bindingdb_examples.tsv", "r") as f:
    sequence_mapping = f.readlines()[1:]
sequence_mapping = [line.split("\t") for line in sequence_mapping]
sequence_mapping = [[line[37], line[38].split(",")[0]] for line in sequence_mapping]
sequence_mapping = dict(zip([line[0] for line in sequence_mapping],
                            [line[1] for line in sequence_mapping]))

