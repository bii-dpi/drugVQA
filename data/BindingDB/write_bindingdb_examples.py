import os
import pickle

import numpy as np
import pandas as pd

from urllib.request import urlopen

NUM_SELECTED = 50

def get_dict(fname):
    with open(fname, "r") as f:
        lines = f.readlines()[1:]
    return dict(zip([lines[i].strip("\n") for i in range(0, len(lines), 2)],
                    [lines[i].strip("\n") for i in range(1, len(lines), 2)]))

def is_dude(name):
    for key in dude_dict.keys():
        if name in key:
            return True
    return False

def get_sequence(name):
    for key in bindingdb_dict.keys():
        if name in key:
            return bindingdb_dict[key]
    raise Exception(f"{name} not in BindingDB dict.")

def get_interaction(nm):
    try:
        nm = float(nm)
        return "1" if nm <= 10 else "0"
    except:
        if nm.startswith(">"):
            return "0"
        elif nm.startswith("<"):
            return "1"
        else:
            raise Exception(f"Unknown nM: {nm}")

bindingdb_dict = get_dict("mapped_bindingdb_sequences.txt")
dude_dict = get_dict("mapped_dude_sequences.txt")

if "sim_matrix.pkl" not in os.listdir():
    response = urlopen(
        "https://www.ebi.ac.uk/Tools/services/rest/clustalo/result/clustalo-E20210616-172123-0663-50769114-p2m/pim"
    )
    sim_matrix = response.read().decode("utf-8")
    with open("sim_matrix.pkl", "wb") as f:
        pickle.dump(sim_matrix, f)
else:
    sim_matrix = pd.read_pickle("sim_matrix.pkl")

matrix = [line.split() for line in sim_matrix.split("\n")[6:-1]]
names = [line[1] for line in matrix]
matrix = np.array([line[2:] for line in matrix
                    if is_dude(line[1])], dtype=float)

sim_mean = np.nanmean(matrix, axis=0)


selected_names = np.array(names)[np.argsort(sim_mean)].tolist()
selected_names = [name for name in selected_names
                    if not is_dude(name)][:NUM_SELECTED]
selected_sequences = [get_sequence(name) for name in selected_names]
print(f"Sim %: {np.sort(sim_mean)[:NUM_SELECTED+5]}")

# Filter the examples, and prepare them.
with open("bindingdb_examples.csv", "r") as f:
    examples = [line.split(",") for line in f.readlines()]

examples = [[element for element in line if element][:3]
            for line in examples[1:]]
examples = [" ".join([line[0], line[2], get_interaction(line[1])]) for
            line in examples
            if line[2] in selected_sequences]
print(f"Num examples: {len(examples)}")
print(f"Actives to decoys: {np.mean([int(line.split()[2]) for line in examples])}")

# Write the prepared file.
with open(f"bindingdb_examples_filtered_{NUM_SELECTED}", "w") as f:
    f.write("\n".join(examples))

