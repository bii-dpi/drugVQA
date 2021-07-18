import pickle

import numpy as np
import pandas as pd

from rdkit import Chem
from progressbar import progressbar
from sklearn.cluster import MiniBatchKMeans
from concurrent.futures import ThreadPoolExecutor


if __name__ == "__main__":
    np.random.seed(12345)

    # Load clusterer model.
    clusterer = MiniBatchKMeans(n_clusters=10000, batch_size=10000,
                                max_iter=1,
                                compute_labels=False, random_state=12345,
                                init=np.load("results/cluster_centers_6400000.npy"))
    clusterer.fit(np.load("results/cluster_centers_6400000.npy"))
    print("Loaded model")

    # Load decoy fingerprints
    with open("data/all_zinc.smi", "r") as f:
        compounds = f.readlines()
    compounds = [line.split()[0] for line in compounds]
    with open("results/zinc_smiles.pkl", "wb") as f:
        pickle.dump(compounds, f)
    print("Saved SMILES")

    with ThreadPoolExecutor() as executor:
        compounds = executor.map(Chem.MolFromSmiles, compounds)
        print("Loaded compounds")
        fingerprints = list(executor.map(Chem.RDKFingerprint, compounds))
        print("Loaded fingerprints")

    with open("results/compounds.pkl", "wb") as f:
        pickle.dump(list(compounds), f)
    del compounds
    print("Pickled compounds")

    with open("results/fingerprints.pkl", "wb") as f:
        pickle.dump(fingerprints, f)
    print("Pickled fingerprints")

    labels = clusterer.predict(fingerprints)
    del fingerprints

    np.save("results/labels.npy", labels)
    print("Pickled labels")

