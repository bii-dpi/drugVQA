import os
import pickle

import numpy as np
import pandas as pd
import deepchem as dc

from progressbar import progressbar
from concurrent.futures import ThreadPoolExecutor


def get_sequence(fname):
    with open(fname, "r") as f:
        return f.readlines()[1].strip("\n")


def load_protein_features(fname):
    with open(fname, "r") as f:
        text = f.readlines()[1:]
    text = [line.split() for line in text]
    return {line[0]: line[1:] for line in text}


def get_ligand_features(features):
    return ligand_featurizer.featurize(features).tolist()[0]


def get_pdb_id(sequence):
    try:
        return sequence_to_id[sequence]
    except:
        return bindingdb_contact_dict[sequence]


def get_protein_features(sequence):
    pdb_id = get_pdb_id(sequence)
    for key in bindingdb_features.keys():
        if pdb_id in key:
            return bindingdb_features[key]
    for key in dude_features.keys():
        if pdb_id in key:
            return dude_features[key]


def standardize(features, mean, std, to_exclude):
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - mean[i]) / std[i]
    features = np.delete(features, to_exclude, axis=1)
    return features


def process_example(example):
    return [get_ligand_features(example[0]) +
            get_protein_features(example[1]),
            example[2]]


def save_features(path):
    global mean, std, to_exclude

    print(f"Processing {path}")
    # Load examples.
    with open(path, "r") as f:
        examples = f.readlines()
    examples = [example.split() for example in examples]
    # Compute features.
    new_examples = []
    with ThreadPoolExecutor() as executor:
        examples = executor.map(process_example, examples)

    # Standardization features.
    features = np.array([example[0] for example in examples],
                        dtype=float)
    if mean is None:
        mean = features.mean(0)
        std = features.std(0)
        to_exclude = np.where(mean == 0)[0]
    features = standardize(features, mean, std, to_exclude)
    labels = np.array([example[1] for example in examples],
                      dtype=float)

    # Save
    np.save(f"{path}_features.npy", features)
    np.save(f"{path}_labels.npy", labels)

    return features, labels


example_paths = ["shallow_training_examples",
                 "shallow_validation_examples",
                 "shallow_testing_examples"]

ligand_featurizer = dc.feat.RDKitDescriptors()

dude_cmaps = [fname for fname in os.listdir("../DUDE/contact_map/")
                if fname.endswith("_full")]
bindingdb_cmaps = [fname for fname in os.listdir("../BindingDB/contact_map/")
                   if fname.endswith("_cm")]

dude_sequences = [get_sequence(f"../DUDE/contact_map/{fname}") for fname in dude_cmaps]
bindingdb_sequences = [get_sequence(f"../BindingDB/contact_map/{fname}") for fname in bindingdb_cmaps]

dude_pdb_ids = [fname.split("_")[1][:-1].upper() for fname in dude_cmaps]
bindingdb_pdb_ids = [fname.split("_")[0] for fname in bindingdb_cmaps]

sequence_to_id = dict(zip(dude_sequences + bindingdb_sequences,
                          dude_pdb_ids + bindingdb_pdb_ids))
with open("../BindingDB/contact_map/BindingDB-contactDict", "r") as f:
    bindingdb_contact_dict = [line.split(":") for line in f.readlines()]
    bindingdb_contact_dict = {line[0]: line[1].strip("\n").split("_cm")[0] for line in bindingdb_contact_dict}

"""
with open("../contact_map/BindingDB-contactDict", "r") as f:
sequence_to_id_dict = f.readlines()

sequence_to_id_dict = {line.split(":")[0]:line.split(":")[1].strip("\n")
for line in sequence_to_id_dict}
"""

with open("shallow_testing_examples", "r") as f:
    proteins = np.unique([line.split()[1] for line in f.readlines()])
#print(len([protein for protein in proteins
#           if protein in bindingdb_sequences]))

bindingdb_features = load_protein_features("bindingdb_protein_features")
dude_features = load_protein_features("dude_protein_features")

if __name__ == "__main__":
    mean, std, to_exclude = None, None, None
    for path in example_paths:
        features, labels = save_features(path)

