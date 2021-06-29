import numpy as np
import pandas as pd

from os import listdir


def has_cm(target_sequence):
    return target_sequence in sequence_to_id_dict.keys()


def get_affinity(i):
    if not pd.isna(bindingdb_examples.ki.iloc[i]):
        nm = bindingdb_examples.ki.iloc[i]
    elif not pd.isna(bindingdb_examples.ic50.iloc[i]):
        nm = bindingdb_examples.ic50.iloc[i]
    elif not pd.isna(bindingdb_examples.kd.iloc[i]):
        nm = bindingdb_examples.kd.iloc[i]
    elif not pd.isna(bindingdb_examples.ec50.iloc[i]):
        nm = bindingdb_examples.ec50.iloc[i]
    else:
        nm = ""
    try:
        return float(nm) / 1000
    except:
        try:
            return float(nm[1:]) / 1000
        except:
            return None


bindingdb_examples = pd.read_csv("bindingdb_examples.tsv", sep="\t")

with open("../contact_map/BindingDB-contactDict", "r") as f:
    sequence_to_id_dict = f.readlines()

sequence_to_id_dict = {line.split(":")[0]:line.split(":")[1].strip("\n")
                        for line in sequence_to_id_dict}

selected_sequences = [target_sequence for target_sequence in
                        bindingdb_examples.target_sequence if
                        has_cm(target_sequence)]

bindingdb_examples = bindingdb_examples[
                                        bindingdb_examples.\
                                        target_sequence.\
                                        isin(selected_sequences)
                                       ]

affinities = [get_affinity(i) for i in range(len(bindingdb_examples))]
affinities = np.array([affinity for affinity in affinities
                        if affinity is not None])

print(f"Actives: {np.mean(affinities <= 1)}")
for cutoff in range(1, 6):
      print(f"Inactives: {np.mean(affinities >= cutoff * 10)}")

