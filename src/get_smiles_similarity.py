import numpy as np


def get_smiles_strings(fold_num):
    with open(f"../data/DUDE/data_pre/rv_{fold_num}_val_fold", "r") as f:
        smiles_strings = f.readlines()
    return [line.split()[0] for line in smiles_strings]


dude_smiles_strings = []
for fold_num in range(1, 4):
    dude_smiles_strings += get_smiles_strings(fold_num)

dude_smiles_strings = np.unique(dude_smiles_strings).tolist()

with open("../data/BindingDB/data_pre/bindingdb_examples_filtered_-1", "r") as f:
    bindingdb_smiles_strings = f.readlines()
bindingdb_smiles_strings = [line.split()[0] for line in
                            bindingdb_smiles_strings]

print(np.mean([int(smiles_string in dude_smiles_strings)
               for smiles_string in bindingdb_smiles_strings]))
