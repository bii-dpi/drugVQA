import numpy as np
from protein import *


NUM_PROTEINS = -1
NUM_EXAMPLES = -1
SHUFFLE_SEED

# Load proteins
bindingdb_proteins = [Protein(name) for name in list(bindingdb_dict.keys())
                      if Protein.cm_exists(name)]
bindingdb_proteins = [protein for protein in bindingdb_proteins
                      if len(protein.get_examples()) >= 50]

# Select BindingDB proteins to keep.
dude_sim_means = [protein.get_dude_sim_mean() for protein in bindingdb_proteins]
print(np.round(sorted(dude_sim_means), 2))

selected_proteins = \
    np.array([protein for protein in bindingdb_proteins])\
    [np.argsort(dude_sim_means)][:NUM_PROTEINS]

# Add computational inactives.
[protein.add_inactives(bindingdb_proteins) for protein in selected_proteins]

# Keep only the proteins that have enough inactives.
selected_proteins = [protein for protein in selected_proteins
                     if protein.get_ratio() >= 50]

[protein.resample_inactives() for protein in selected_proteins]

# Compile all selected examples.
all_examples = []
for protein in selected_proteins:
    all_examples += protein.get_examples()

np.random.shuffle(all_examples)
all_examples = all_examples[:NUM_EXAMPLES]

with open(f"new_bindingdb_examples_{NUM_PROTEINS}", "w") as f:
    f.write("\n".join(all_examples))

