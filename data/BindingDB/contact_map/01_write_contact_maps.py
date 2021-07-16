import os
import numpy as np
import pandas as pd

from Bio.PDB import *
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor


p = PDBParser(PERMISSIVE=1)


def get_pdb_file(pdb_id):
    try:
        return p.get_structure(pdb_id,
                               f"pdb_files/pdb{pdb_id.lower()}.ent")
    except:
        raise Exception(f"{pdb_id} has no file.")


def get_coord_list(structure):
    coord_list = []

    for model in structure.get_models():
        for chain in model:
            for residue in chain:
                if residue.get_full_id()[-1][0] != " ":
                    continue
                for atom in residue:
                    if atom.get_name().startswith("CA"):
                        coord_list.append(atom.get_coord())
                        break
        break

    return np.array(coord_list)


def get_distance(coords_1, coords_2):
    return 1 / (1 + (np.linalg.norm(coords_1 - coords_2) / 3.8))


def get_distance_matrix(coord_list):
    dist_matrix = np.zeros((len(coord_list), len(coord_list)))
    for i in range(len(coord_list)):
        for j in range(len(coord_list)):
            dist_matrix[i, j] = get_distance(coord_list[i],
                                             coord_list[j])
    return dist_matrix


def get_matrix_string(pdb_id, sequence, dist_matrix):
    text = f"{pdb_id}\n{sequence}"
    for i in range(dist_matrix.shape[0]):
        text += "\n"
        for j in range(dist_matrix.shape[0]):
            text += f"{dist_matrix[i, j]} "
    return text


def save_contact_map(sequence):
    pdb_id = sequence_mapping[sequence]
    if f"{pdb_id}_cm" in os.listdir():
       return None
    structure = get_pdb_file(pdb_id)
    coord_list = get_coord_list(structure)
    dist_matrix = get_distance_matrix(coord_list)
    text = get_matrix_string(pdb_id, sequence, dist_matrix)
    with open(f"{pdb_id}_cm", "w") as f:
        f.write(text)


# Get BindingDB sequences and PDB IDs.
sequence_mapping = pd.read_pickle("sequence_to_id_map.pkl")
bindingdb_sequences = sequence_mapping.keys()
bindingdb_sequences = [sequence for sequence in bindingdb_sequences
                        if sequence_mapping[sequence] + "_cm" not in
                        os.listdir()]
print(len(bindingdb_sequences))

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        results = executor.map(save_contact_map, bindingdb_sequences)
"""
for sequence in progressbar(bindingdb_sequences):
    save_contact_map(sequence)
"""
