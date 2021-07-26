import os
import pickle

import numpy as np
import pandas as pd

from progressbar import progressbar
from rdkit import Chem, DataStructs
from concurrent.futures import ProcessPoolExecutor


ILLEGAL_LIST = ["[c-]", "[N@@]", "[Re-]", "[S@@+]", "[S@+]"]


def get_smiles(label):
    return zinc_smiles[np.where(labels == label)[0]]


def get_fingerprint(mol):
    try:
        return Chem.RDKFingerprint(mol)
    except:
        return None


def has_illegal(smiles):
    for illegal_element in ILLEGAL_LIST:
        if illegal_element in smiles:
            return True
    return False


def get_max_sim(candidate_smiles, active_fingerprints, threshold):
    try:
        candidate_fingerprint = \
            Chem.RDKFingerprint(Chem.MolFromSmiles(candidate_smiles))

        return np.max([DataStructs.FingerprintSimilarity(candidate_fingerprint,
                                                         active_fingerprint)
                       for active_fingerprint in active_fingerprints])
    except:
        return threshold + 1


def write_decoys(protein_id, decoy_smiles):
    with open(f"results/{protein_id}_decoys", "w") as f:
        f.write("\n".join(decoy_smiles))


def get_decoys_smiles(protein_id, threshold=0.5):
    active_smiles = active_smiles_dict[protein_id]
    active_mols = [Chem.MolFromSmiles(smiles)
                   for smiles in active_smiles]
    active_fingerprints = [get_fingerprint(mol) for mol in active_mols]
    active_fingerprints = [fingerprint for fingerprint in active_fingerprints
                           if fingerprint]

    decoy_smiles = []
    while len(decoy_smiles) < len(active_smiles) * 50:
        candidate_smiles = \
            np.random.choice(sampled_smiles,
                             len(active_smiles) * 50 - len(decoy_smiles))

        candidate_smiles = [smiles for smiles in candidate_smiles
                            if smiles not in decoy_smiles
                            and not has_illegal(smiles)
                            and get_max_sim(smiles,
                                            active_fingerprints,
                                            threshold) <= threshold]
        decoy_smiles += candidate_smiles

    write_decoys(protein_id, decoy_smiles)


if __name__ == "__main__":
    np.random.seed(12345)

    print("Loading SMILES and corr. labels...")
    zinc_smiles = np.array(pd.read_pickle("results/zinc_smiles.pkl"))
    labels = np.load("results/labels.npy")

    print("Loading decoy dict...")
    if "decoy_dict.pkl" not in os.listdir("results/"):
        with ThreadPoolExecutor() as executor:
            label_lists = executor.map(get_smiles, range(10000))
        decoy_dict = dict(zip(range(10000), label_lists))
        with open("results/decoy_dict.pkl", "wb") as f:
            pickle.dump(decoy_dict, f)
    else:
        decoy_dict = pd.read_pickle("results/decoy_dict.pkl")

    print("Sampling decoy dict...")
    sampled_dict = {label: np.random.choice(decoy_dict[label], 40)
                    for label in range(10000)}

    # Where to take decoys from.
    sampled_smiles = [element for label in range(10000)
                      for element in sampled_dict[label]]

    del zinc_smiles, labels, decoy_dict, sampled_dict

    # Load actives data.
    with open("data/all_actives", "r") as f:
        all_active_smiles = f.readlines()
    all_active_smiles = [line.split() for line in all_active_smiles]
    all_protein_ids = np.unique([line[1] for line in all_active_smiles])

    active_smiles_dict = {}
    for protein_id in all_protein_ids:
        active_smiles_dict[protein_id] = [line[0] for line in all_active_smiles
                                          if line[1] == protein_id]
    del all_active_smiles

    """
    for protein_id in progressbar(all_protein_ids):
        get_decoys_smiles(protein_id)
    """
    with ProcessPoolExecutor() as executor:
        executor.map(get_decoys_smiles, all_protein_ids)

    print("Written all decoys.")

