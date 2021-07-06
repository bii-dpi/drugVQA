import pickle

import numpy as np
import deepchem as dc


def get_dict(fname):
    with open(f"../BindingDB/data_pre/{fname}.txt", "r") as f:
        text = f.readlines()[1:]
    return {text[i + 1].strip("\n"): text[i].strip("\n")
            for i in range(0, len(text), 2)}


def load_protein_features(fname):
    with open(fname, "r") as f:
        text = f.readlines()[1:]
    text = [line.split() for line in text]
    return {line[0]: line[1:] for line in text}


def get_ligand_features(features):
    return ligand_featurizer.featurize(features).tolist()[0]


def get_pdb_id(sequence):
    try:
        return bindingdb_to_id[sequence]
    except:
        return dude_to_id[sequence]


def get_protein_features(sequence):
    pdb_id = get_pdb_id(sequence)
    try:
        for key in bindingdb_features.keys():
            if f"{key} " in pdb_id:
                return bindingdb_features[key]
    except:
        for key in dude_features.keys():
            if f"{key} " in pdb_id:
                return dude_features[key]


def standardize(features, mean, std):
    for i in range(features.shape[1]):
        print(mean[i], std[i])
        features[:, i] = (features[:, i] - mean[i]) / std[i]
    return features


def save_features(path):
    global mean, std

    print(f"Processing {path}")
    # Load examples.
    with open(path, "r") as f:
        examples = f.readlines()
    examples = [example.split() for example in examples][:1000]
    print(f"Num examples before: {len(examples)}")
    # Compute features.
    examples = [[get_ligand_features(example[0]) +
                 get_protein_features(example[1]),
                 example[2]] for example in examples]
    print(examples[1])

    """
    examples = [example for example in examples
                if example[1] is not None]
    """
    print(f"Num examples after: {len(examples)}")

    # Standardization features.
    features = np.array([example[0] for example in examples],
                        dtype=float)
    if mean is None:
        mean = features.mean(0)
        std = features.std(0)
        # Check if any mean is 0
    features = standardize(features, mean, std)
    labels = np.array([example[1] for example in examples],
                      dtype=float)

    # Save
    np.save(f"{path}_features.npy", features)
    np.save(f"{path}_labels.npy", labels)

    return features, labels


example_paths = ["shallow_training_examples",
                 "shallow_validation_examples",
                 "shallow_testing_examples"][-1:]

ligand_featurizer = dc.feat.RDKitDescriptors()

bindingdb_to_id = get_dict("mapped_bindingdb_sequences")
dude_to_id = get_dict("mapped_dude_sequences")
bindingdb_features = load_protein_features("bindingdb_protein_features")
dude_features = load_protein_features("dude_protein_features")

print(dude_to_id)

mean, std = None, None
for path in example_paths:
    features, labels = save_features(path)
print(features)
print(labels)
