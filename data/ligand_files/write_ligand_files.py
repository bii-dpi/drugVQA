import os
import pickle

import numpy as np

from rdkit import Chem
from progressbar import progressbar


def get_smiles_strings(path):
    with open(path, "r") as f:
        examples = f.readlines()
    return np.unique([example.split()[0] for example in examples])


def write_sdf_file(smiles_string, i):
    mol = Chem.MolFromSmiles(smiles_string)
    with Chem.rdmolfiles.SDWriter(f"{i}.mol") as w:
        w.write(mol)


if "smiles_dict.pkl" in os.listdir():
    with open("smiles_dict.pkl", "rb") as f:
        smiles_dict = pickle.load(f)
else:
    smiles_dict = {}


all_smiles_strings = [element for path in
                      ["../DUDE/data_pre/shallow_training_examples",
                       "../DUDE/data_pre/shallow_validation_examples",
                       "../BindingDB/data_pre/shallow_testing_examples"]
                      for element in get_smiles_strings(path)]

strings_to_do = [smiles_string for smiles_string in all_smiles_strings
                 if smiles_string not in smiles_dict.values()]


i = 0
if smiles_dict.keys():
    i = max(list(smiles_dict.keys())) + 1


for smiles_string in progressbar(strings_to_do):
    write_sdf_file(smiles_string, i)
    smiles_dict[i] = smiles_string
    i += 1

with open("smiles_dict.pkl", "wb") as f:
    pickle.dump(smiles_dict, f)

