import pickle

import deepchem as dc


ligand_featurizer = dc.feat.RDKitDescriptors()


def get_ligand_features(features):
    return ligand_featurizer.featurize(features)


def get_protein_features(sequence):



def get_featues(path):
    print(f"Processing {path}")
    with open(path, "r") as f:
        examples = f.readlines()
    examples = [example.split() for example in examples]
    print(f"Num examples before: {len(examples)}")
    examples = [[get_ligand_features(example[0]),
                 get_protein_features(example[1]),
                 example[2]] for example in examples]
    examples = [example for example in examples
                if None not in example]
    print(f"Num examples after: {len(examples)}")
    return examples


example_paths = ["../DUDE/data_pre/shallow_training_examples",
                 "../DUDE/data_pre/shallow_validation_examples",
                 "../BindingDB/data_pre/shallow_testing_examples"]

print([featurizer.featurize(get_paths(path)) for path in example_paths])

