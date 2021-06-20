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
        if nm.startswith(">") or nm.startswith("<"):
            return get_interaction(nm[1:])
        else:
            raise Exception(f"Unknown nM: {nm}")


def create_example(i):
    line = (f"{bindingdb_examples.ligand_smiles.iloc[i]} "
            f"{bindingdb_examples.target_sequence.iloc[i]}")
    if not pd.isna(bindingdb_examples.ki.iloc[i]):
        nm = bindingdb_examples.ki.iloc[i]
    elif not pd.isna(bindingdb_examples.ic50.iloc[i]):
        nm = bindingdb_examples.ic50.iloc[i]
    elif not pd.isna(bindingdb_examples.kd.iloc[i]):
        nm = bindingdb_examples.kd.iloc[i]
    elif not pd.isna(bindingdb_examples.ec50.iloc[i]):
        nm = bindingdb_examples.ec50.iloc[i]
    try:
        return f"{line} {get_interaction(nm)}"
    except:
#        print(f"{i} has no valid int.")
        return ""

bindingdb_dict = get_dict("mapped_bindingdb_sequences.txt")
dude_dict = get_dict("mapped_dude_sequences.txt")

if "sim_matrix.pkl" not in os.listdir():
    response = urlopen(
        "https://www.ebi.ac.uk/Tools/services/rest/clustalo/result/clustalo-I20210619-205247-0024-85657637-p2m/pim"
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
bindingdb_examples = pd.read_csv("bindingdb_examples.tsv", sep="\t")
bindingdb_examples = bindingdb_examples[
                                        bindingdb_examples.\
                                        target_sequence.\
                                        isin(selected_sequences)
                                       ]

filtered_examples = []
for i in range(len(bindingdb_examples)):
    filtered_examples.append(create_example(i))

filtered_examples = [example for example in filtered_examples if example]

print(f"Num examples: {len(filtered_examples)}")
print(f"Actives to decoys: {np.mean([int(line.split()[2]) for line in filtered_examples])}")

# Write the prepared file.
with open(f"bindingdb_examples_filtered_{NUM_SELECTED}", "w") as f:
    f.write("\n".join(filtered_examples))



