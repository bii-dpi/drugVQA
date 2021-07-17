import pickle

import numpy as np

from protein import *


NUM_PROTEINS = -1
NUM_EXAMPLES = -1
SHUFFLE_SEED = 12345


# Load proteins
bindingdb_proteins = [Protein(name) for name in list(bindingdb_dict.keys())
                      if Protein.cm_exists(name)]
bindingdb_proteins = [protein for protein in bindingdb_proteins
                      if len(protein.get_examples()) >= 40]

# Select BindingDB proteins to keep.
dude_sim_means = [protein.get_dude_sim_mean() for protein in bindingdb_proteins]
print(np.round(sorted(dude_sim_means), 2))

selected_proteins = \
    np.array([protein for protein in bindingdb_proteins])\
    [np.argsort(dude_sim_means)][:NUM_PROTEINS]

# Get selected protein similarities.
sims = [protein.get_sims(selected_proteins) for protein in selected_proteins]
sims = [element for sublist in sims for element in sublist]

# Write active SMILES and corr. protein PDB IDs.
all_actives = []
for protein in selected_proteins:
    curr_actives = protein.get_actives()
    all_actives += [f"{line.split()[0]} {protein.get_id()}" for line in curr_actives]

np.random.shuffle(all_actives)
all_actives = all_actives[:NUM_EXAMPLES]

with open(f"all_actives.txt", "w") as f:
    f.write("\n".join(all_actives))

with open(f"../../../../clustering/data/all_actives.txt", "w") as f:
    f.write("\n".join(all_actives))

# Write SMILES to ZINC ID dictionary.
smiles_to_zinc = pd.read_csv("bindingdb_examples.tsv", sep="\t")
smiles_to_zinc = dict(zip(smiles_to_zinc.ligand_smiles.tolist(),
                          smiles_to_zinc.ligand_zinc_id.tolist()))

with open("smiles_to_zinc.pkl", "wb") as f:
    pickle.dump(smiles_to_zinc, f)

with open("../../../../clustering/data/smiles_to_zinc.pkl", "wb") as f:
    pickle.dump(smiles_to_zinc, f)


ILLEGAL_LIST = ["[c-]", "[N@@]", "[Re-]"]


def has_illegal(line):
    smiles_string = line.split()[0]
    for illegal_element in ILLEGAL_LIST:
        if illegal_element in smiles_string:
            return True
    return False


with open(f"bindingdb_examples", "r") as f:
    text = f.readlines()

print(len(text))
text = [line for line in text if not has_illegal(line)]
print(len(text))

with open(f"bindingdb_examples", "w") as f:
    f.writelines(text)

