import pickle

import numpy as np

from rdkit import Chem
from progressbar import progressbar
from sklearn.cluster import MiniBatchKMeans
from concurrent.futures import ThreadPoolExecutor


BATCH_SIZE = 100000

if __name__ == "__main__":
    np.random.seed(12345)

    clusterer = MiniBatchKMeans(n_clusters=10000, batch_size=BATCH_SIZE,
                                compute_labels=False, random_state=12345)

    with open("data/all_zinc.smi", "r") as f:
        compounds = f.readlines()
    compounds = [line.split()[0] for line in compounds]

    with ThreadPoolExecutor() as executor:
        compounds = executor.map(Chem.MolFromSmiles, compounds)
        fingerprints = list(executor.map(Chem.RDKFingerprint, compounds))
    del compounds

    np.random.shuffle(fingerprints)

    indices = list(range(0, len(fingerprints), BATCH_SIZE)) + [-1]

    for i in progressbar(range(len(indices) - 1)):
        clusterer.partial_fit(fingerprints[indices[i]:indices[i + 1]])

        np.save(f"results/cluster_centers_{indices[i]}.npy", clusterer.cluster_centers_)

