import numpy as np

from protein import *


NUM_PROTEINS = -1
NUM_EXAMPLES = 5000


# Load proteins
bindingdb_proteins = [Protein(pdb_id) for pdb_id in list(bindingdb_dict.keys())]

# Select BindingDB proteins to keep.
dude_sim_means = [protein.get_dude_sim_mean() for protein in bindingdb_proteins]

bindingdb_proteins = np.array(bindingdb_proteins)[np.argsort(dude_sim_means)].tolist()

if NUM_PROTEINS == -1:
    NUM_PROTEINS = len(bindingdb_proteins)

selected_proteins = []
while len(selected_proteins) < NUM_PROTEINS and bindingdb_proteins:
    candidate_protein = bindingdb_proteins.pop(0)
    accept = True
    for existing_protein in selected_proteins:
        if existing_protein.get_sim(candidate_protein.get_pdb_id()) > 20:
            accept = False
            break
    if accept:
        selected_proteins.append(candidate_protein)

dude_sim_means = dude_sim_means[:NUM_PROTEINS]
print("DUDE similarity statistics")
print(f"Mean: {np.nanmean(dude_sim_means):.2f}, std.: "
      f"{np.nanstd(dude_sim_means):.2f}, max: {max(dude_sim_means)}, "
      f"min: {min(dude_sim_means)}")


sims = [protein.get_sims(selected_proteins) for protein in selected_proteins]
sims = [element for sublist in sims for element in sublist]
print("Intra similarity statistics")
print(f"Mean: {np.nanmean(sims):.2f}, std.: {np.nanstd(sims):.2f}, max: {max(sims)}, min: {min(sims)}")


# Write active SMILES and corr. protein PDB IDs.
all_actives = []
for protein in selected_proteins:
    curr_actives = protein.get_actives()
    all_actives += [f"{line.split()[0]} {protein.get_pdb_id()}" for line in curr_actives]

np.random.shuffle(all_actives)
all_actives = all_actives[:NUM_EXAMPLES]
print(len(all_actives))

with open(f"../../../../clustering/data/all_actives", "w") as f:
    f.write("\n".join(all_actives))

