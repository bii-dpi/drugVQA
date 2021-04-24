"""Create training and validation folds."""

import pickle
import numpy as np
import pandas as pd


# Application
target_names = \
    pd.read_csv(f"data/DUDE/dud-e_proteins_reduced.csv")["name"]
target_names = [target_name.lower() for target_name in target_names]


# Load sequence-protein mapping dictionary.
with open(f"data/DUDE/dataPre/DUDE-contactDict") as f:
    contact_dict = [line.split(":") for line in f.readlines()]
    contact_dict = dict([[line[0], line[1].split("_")[0]] for
                            line in contact_dict])
print(sorted(list(set(contact_dict.values()))))


# Load all examples.
for i in range(1, 4):
    all_examples = []
    with open(f"data/DUDE/dataPre/DUDE-foldTrain{i}") as f:
        all_examples += f.readlines()

all_examples = list(set(all_examples))
all_examples = [line.split() for line in all_examples]


# Map each example to its target.
all_examples_dict = {}
for target_name in target_names:
    all_examples_dict[target_name] = [line for line in all_examples
                                        if contact_dict[line[1]] == target_name]


for key, item in all_examples_dict.items():
    print(f"{key}: {len(item)}")
