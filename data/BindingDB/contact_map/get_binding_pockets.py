import os
import pickle
import numpy as np

from Bio.PDB import *
from glob import glob
from progressbar import progressbar


p = PDBParser(PERMISSIVE=1)


def get_pocket_coords(pdb_id):
    coord_list = []
    structure = p.get_structure("X", f"pdb_files/pdb{pdb_id}.ent")
    for model in structure.get_models():
        for chain in model:
            for residue in chain:
                if residue.get_full_id()[-1][0].startswith(" "):
                    continue
                for atom in residue:
                    coord_list.append(atom.get_coord())
        break

    coord_list = np.array(coord_list)
    try:
        maxes = np.amax(coord_list, 0)
        mins = np.amin(coord_list, 0)
    except:
        print(coord_list.shape)
    return coord_list


def write_binding_pocket_coords(pdb_id):
    with open(f"{pdb_id.upper()}_pocket_coords.pkl", "wb") as f:
        pickle.dump(get_pocket_coords(pdb_id), f)


pdb_ids = [fname.split("_")[0].lower() for fname in os.listdir()
              if fname.endswith("_cm")]

for pdb_id in progressbar(pdb_ids):
    write_binding_pocket_coords(pdb_id)

