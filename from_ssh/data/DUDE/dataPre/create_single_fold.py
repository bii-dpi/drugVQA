"""Create training and validation folds."""

import pickle
import numpy as np
import pandas as pd


TRAINING_RESAMPLE_TO = 1
VALIDATION_PROP = 0.2
SEED = 12345

np.random.seed(SEED)


# Helper functions
def get_examples(target, interactivity_type):
    """Get `interactivity_type` examples for `target`."""
    with open(f"../{interactivity_type}_smile/"
                f"{target}_{interactivity_type}_final.ism") as f:
        curr_examples  = [" ".join([line.split()[0],
                            contact_dict[target],
                            str(int(interactivity_type == "actives"))])
                            for line in f.readlines()]
    np.random.shuffle(curr_examples)
    return curr_examples


def write_shuffled_data(example_dict, mode):
    """Write shuffled list of active and decoy examples. """
    examples = example_dict[mode]["actives"] + example_dict[mode]["decoys"]
    np.random.shuffle(examples)
    with open(f"single_{mode}_fold", "w") as f:
        f.write("\n".join(examples))


# Application
# Load all 102 target names.
all_targets = \
    pd.read_csv(f"dud-e_proteins.csv")["name"]
all_targets = [target.lower() for target in all_targets]


# Load target name-sequence mapping dictionary.
with open(f"DUDE-contactDict") as f:
    contact_dict = [line.split(":") for line in f.readlines()]
    contact_dict = dict([[line[1].split("_")[0], line[0]] for
                            line in contact_dict])


# Populate the dictionary with all examples.
example_dict = {"training": {"actives": [], "decoys": []},
                    "validation": {"actives": [], "decoys": []}}
for target in all_targets:
    curr_actives = get_examples(target, "actives")
    curr_decoys = get_examples(target, "decoys")

    num_training_actives = int(len(curr_actives) * (1 - VALIDATION_PROP))

    example_dict["training"]["actives"] += \
        curr_actives[:num_training_actives]
    example_dict["validation"]["actives"] += \
        curr_actives[num_training_actives:]

    example_dict["training"]["decoys"] += \
        curr_decoys[:int(num_training_actives * TRAINING_RESAMPLE_TO)]

    # Keep 50 times the number of validation decoys as actives.
    decoys_remaining = curr_decoys[int(num_training_actives * TRAINING_RESAMPLE_TO):]
    num_decoys_needed = (len(curr_actives) - num_training_actives) * 50
    if len(decoys_remaining) < num_decoys_needed:
        num_decoys_needed = -1
    example_dict["validation"]["decoys"] += \
        decoys_remaining[:num_decoys_needed]


print(f"Training ratio: {len(example_dict['training']['decoys']) / len(example_dict['training']['actives']):.2f}")
print(f"Validation ratio: {len(example_dict['validation']['decoys']) / len(example_dict['validation']['actives']):.2f}")


for key_1 in ["training", "validation"]:
    for key_2 in ["actives", "decoys"]:
        np.random.shuffle(example_dict[key_1][key_2])


# Integrate actives and decoys, shuffle them, and then write the folds
write_shuffled_data(example_dict, "training")
write_shuffled_data(example_dict, "validation")
