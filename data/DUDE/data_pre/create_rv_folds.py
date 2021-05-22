"""Create training and validation folds."""

import pickle
import numpy as np
import pandas as pd


TRAINING_RESAMPLE_TO = 1
VALIDATION_PROP = 0.2
SEED = 12345

np.random.seed(SEED)


# Helper functions
def write_shuffled_data(example_dict, fname):
    """Write shuffled list of active and decoy examples. """
    examples = example_dict["actives"] + example_dict["decoys"]
    np.random.shuffle(examples)
    with open(fname, "w") as f:
        f.write("\n".join(examples))


def get_examples(target, interactivity_type, contact_dict):
    """Get `interactivity_type` examples for `target`."""
    with open(f"../{interactivity_type}_smile/"
                f"{target}_{interactivity_type}_final.ism") as f:
        curr_examples  = [" ".join([line.split()[0],
                            contact_dict[target],
                            str(int(interactivity_type == "actives"))])
                            for line in f.readlines()]
    np.random.shuffle(curr_examples)
    return curr_examples


def get_cv_val_fold_proteins(fold_index):
    with open(f"orig_val_{fold_index}", "r") as f:
        return [protein_name.split("_")[0] for protein_name
                in f.read().split()]


def get_train_fold_proteins(fold_index):
    return new_train_fold_proteins[fold_index - 1]


def get_val_fold_proteins(fold_index):
    return new_val_fold_proteins[fold_index - 1]


def write_train_fold(fold_index, train_fold_proteins, contact_dict):
    # Populate the dictionary with all examples.
    example_dict = {"actives": [], "decoys": []}
    for target in train_fold_proteins:
        curr_actives = get_examples(target, "actives", contact_dict)
        curr_decoys = get_examples(target, "decoys", contact_dict)

        num_training_actives = len(curr_actives)

        example_dict["actives"] += curr_actives
        example_dict["decoys"] += \
            curr_decoys[:int(num_training_actives * TRAINING_RESAMPLE_TO)]

    print(f"Training ratio: {len(example_dict['decoys']) / len(example_dict['actives']):.2f}")

    for interaction_type in ["actives", "decoys"]:
        np.random.shuffle(example_dict[interaction_type])

    write_shuffled_data(example_dict, f"rv_{fold_index}_train_fold")


def write_val_fold(fold_index, val_fold_proteins, contact_dict):
    # Populate the dictionary with all examples.
    example_dict = {"actives": [], "decoys": []}
    for target in val_fold_proteins:
        curr_actives = get_examples(target, "actives", contact_dict)
        curr_decoys = get_examples(target, "decoys", contact_dict)

        num_actives_needed = len(curr_actives)
        num_decoys_needed = len(curr_actives) * 50

        if num_decoys_needed > len(curr_decoys):
            num_decoys_needed = len(curr_decoys)
            num_actives_needed = int(len(curr_decoys) / 50)

        example_dict["actives"] += curr_actives[:num_actives_needed]
        example_dict["decoys"] += curr_decoys[:num_decoys_needed]

    print(f"Validation ratio: {len(example_dict['decoys']) / len(example_dict['actives']):.2f}")

    for interaction_type in ["actives", "decoys"]:
        np.random.shuffle(example_dict[interaction_type])

    # Integrate actives and decoys, shuffle them, and then write the folds
    write_shuffled_data(example_dict, f"rv_{fold_index}_val_fold")


def create_rv_val_folds(fold_index):
    print(f"Creating CV fold {fold_index}...")
    # Load all 102 target names.
    all_targets = \
        pd.read_csv(f"dud-e_proteins.csv")["name"]
    all_targets = [target.lower() for target in all_targets]

    # Load target name-sequence mapping dictionary.
    with open(f"DUDE-contactDict", "r") as f:
        contact_dict = [line.split(":") for line in f.readlines()]
        contact_dict = dict([[line[1].split("_")[0], line[0]] for
                                line in contact_dict])

    train_fold_proteins = get_train_fold_proteins(fold_index)
    val_fold_proteins = get_val_fold_proteins(fold_index)
    assert not set(train_fold_proteins).intersection(set(val_fold_proteins))

    write_train_fold(fold_index, train_fold_proteins, contact_dict)
    write_val_fold(fold_index, val_fold_proteins, contact_dict)



# Application
orig_val_fold_proteins = {i: get_cv_val_fold_proteins(i) for i in range(1, 4)}

new_val_fold_proteins = {}
new_train_fold_proteins = {}


for i in range(1, 4):
    np.random.shuffle(orig_val_fold_proteins[i])


for i in range(1, 4):
    new_val_fold_proteins[i] = []
    new_train_fold_proteins[i] = []
    for j in range(1, 4):
        curr_proteins = orig_val_fold_proteins[j]
        selected_val_proteins = \
                orig_val_fold_proteins[j][(i - 1) * int(len(orig_val_fold_proteins[j]) / 3):
                                            i * int(len(orig_val_fold_proteins[j]) / 3)]
        new_val_fold_proteins[i] += selected_val_proteins
        new_train_fold_proteins[i] += \
                [protein for protein in curr_proteins
                        if protein not in selected_val_proteins]


print(new_val_fold_proteins)
# Do the checks here within and between folds

"""
create_rv_val_folds(1)
create_rv_val_folds(2)
create_rv_val_folds(3)
"""

