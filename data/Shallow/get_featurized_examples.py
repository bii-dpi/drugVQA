import pickle

import deepchem as dc


# Protein path dictionaries.
with open("../BindingDB/data_pre/mapped_bindingdb_sequences.pkl", "rb") as f:
    sequence_to_id_bindingdb = pickle.load(f)

with open("../BindingDB/data_pre/mapped_dude_sequences.pkl", "rb") as f:
    sequence_to_id_dude = pickle.load(f)


# Ligand path dictionaries.
with open("../ligand_files/smiles_dict.pkl", "rb") as f:
    index_to_smiles = pickle.load(f)
smiles_to_index = {value: key for key, value in index_to_smiles.items()}

print("Loaded dictionaries...")


def get_ligand_path(smiles_string):
    return f"../ligand_files/{smiles_to_index[smiles_string]}.mol"


def get_protein_path(sequence):
    try:
        pdb_id = sequence_to_id_bindingdb[sequence]
        print(pdb_id)
        return "../BindingDB/contact_map/pdb_files/pdb{pdb_id}.ent"
    except:
        pdb_id = sequence_to_id_dude[sequence]
        print(pdb_id)
        return "../DUDE/contact_map/pdb_files/pdb{pdb_id}.ent"


def get_paths(path):
    print(f"Processing {path}")
    with open(path, "r") as f:
        examples = f.readlines()
    examples = [example.split() for example in examples]
    print(f"Num examples before: {len(examples)}")
    examples = [[get_ligand_path(example[0]),
                 get_protein_path(example[1]),
                 example[2]] for example in examples]
    examples = [example for example in examples
                if None not in example]
    print(f"Num examples after: {len(examples)}")
    return examples
    # Do featurization here.


example_paths = ["../DUDE/data_pre/shallow_training_examples",
                 "../DUDE/data_pre/shallow_validation_examples",
                 "../BindingDB/data_pre/shallow_testing_examples"]

featurizer = dc.feat.RdkitGridFeaturizer(feature_types=["flat_combined"])

print(get_paths(example_paths[0]))

print([featurizer.featurize(get_paths(path)) for path in example_paths[:10]])

