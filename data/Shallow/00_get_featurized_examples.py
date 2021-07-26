import os
import pickle

import numpy as np
import pandas as pd
import deepchem as dc

from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor


def get_contactdict(path):
    with open(path, "r") as f:
        contact_dict = [line.split(":") for line in f.readlines()]
    return {line[0]: line[1].strip("\n").split("_cm")[0]
            for line in contact_dict}


def load_protein_features(fname):
    with open(fname, "r") as f:
        text = f.readlines()[1:]
    text = [line.split() for line in text]
    return {line[0]: np.array(line[1:], dtype=float).tolist()
            for line in text}


def get_ligand_features(smiles):
    try:
        return ligand_featurizer.featurize(smiles).tolist()[0]
    except:
        return None


def get_pdb_id(sequence):
    try:
        return dude_contactdict[sequence]
    except:
        return bindingdb_contactdict[sequence]


def get_protein_features(sequence):
    pdb_id = get_pdb_id(sequence)
    try:
        return bindingdb_features[pdb_id]
    except:
        return dude_features[pdb_id]


def process_example(example):
    ligand_features = get_ligand_features(example[0])
    if not ligand_features:
        return []
    protein_features = get_protein_features(example[1])
    return [ligand_features + protein_features,
            example[2]]


def save_features(path):
    print(f"Processing {path}")
    # Load examples.
    with open(path, "r") as f:
        examples = f.readlines()
    examples = [example.split() for example in examples]
    print(f"Examples before: {len(examples)}")

    # Compute features.
    with ProcessPoolExecutor() as executor:
        examples = executor.map(process_example, examples)
    examples = [example for example in examples if example]
    print(f"Examples after: {len(examples)}")

    # Save
    features = np.array([example[0] for example in examples],
                         dtype=float)
    labels = np.array([example[1] for example in examples],
                      dtype=float)

    np.save(f"{path}_features.npy", features)
    np.save(f"{path}_labels.npy", labels)


example_paths = ["dude_examples",
                 "bindingdb_examples"]

ligand_featurizer = dc.feat.RDKitDescriptors()

bindingdb_contactdict = \
    get_contactdict("../BindingDB/contact_map/BindingDB_contactdict")
dude_contactdict = \
    get_contactdict("../DUDE/contact_map/DUDE_contactdict")

bindingdb_features = load_protein_features("bindingdb_protein_features")
dude_features = load_protein_features("dude_protein_features")

if __name__ == "__main__":
    for path in progressbar(example_paths):
        save_features(path)

