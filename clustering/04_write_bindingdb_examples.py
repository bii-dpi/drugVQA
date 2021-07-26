import numpy as np
import pandas as pd

from progressbar import progressbar


np.random.seed(12345)


def get_active_examples(protein_id):
    active_examples = [[line[0].strip().strip("\n"), id_to_sequence[protein_id]]
                       for line in all_active_smiles
                       if line[1] == protein_id]
    return [" ".join(line) + " 1" for line in active_examples]


def get_decoy_examples(protein_id):
    with open(f"results/{protein_id}_decoys", "r") as f:
        decoy_smiles = f.readlines()
    decoy_smiles = [smiles.strip().strip("\n") for smiles in decoy_smiles]
    return [f"{smiles} {id_to_sequence[protein_id]} 0"
            for smiles in decoy_smiles]


def write_examples(protein_id):
    active_examples = get_active_examples(protein_id)
    if len(active_examples) < 10:
        return None
    all_examples = active_examples + \
                    get_decoy_examples(protein_id)
    np.random.shuffle(all_examples)
    with open(f"results/{protein_id}_examples", "w") as f:
        f.write("\n".join(all_examples))


with open("data/all_actives", "r") as f:
    all_active_smiles = f.readlines()
all_active_smiles = [line.split() for line in all_active_smiles]
all_protein_ids = np.unique([line[1] for line in all_active_smiles])

id_to_sequence = pd.read_pickle("../data/BindingDB/contact_map/sequence_to_id_map.pkl")
id_to_sequence = {pdb_id: sequence
                  for sequence, pdb_id in id_to_sequence.items()}

for protein_id in all_protein_ids:
    write_examples(protein_id)

