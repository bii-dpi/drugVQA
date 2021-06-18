import os
import pickle

import numpy as np
import pandas as pd

from urllib.request import urlopen

dir_files = os.listdir()

### Pickle BindingDB sequences.

# Pickle unique BindingDB subset sequences.
if "raw_bindingdb_sequences.pkl" not in dir_files:
    with open("raw_bindingdb_sequences", "r") as f:
        text = f.read().split("\t")

    text = [element.strip().strip("\n").strip("\t")
            for element in text
            if element and
                element.upper() == element and
                "_" not in element]
    raw_bindingdb_sequences = np.unique(text).tolist()

    with open("raw_bindingdb_sequences.pkl", "wb") as f:
        pickle.dump(raw_bindingdb_sequences, f)
else:
    raw_bindingdb_sequences = pd.read_pickle("raw_bindingdb_sequences.pkl")

# Map raw sequences to FASTA sequences.
if "mapped_bindingdb_sequences.pkl" not in dir_files:
    with open("bindingdb_fasta_sequences", "r") as f:
        text = f.readlines()

    fasta_dict = dict(zip([text[i].strip("\n").strip("\t").strip()
                            for i in range(1, len(text), 2)],
                            [text[i].strip("\n")
                                for i in range(0, len(text), 2)]))

    def get_fasta_pair(sequence):
        try:
            return f"\n{fasta_dict[sequence]}\n{sequence}"
        except:
            #print(f"{sequence} not mapped.")
            return None

    mapped_bindingdb_sequences = []
    for sequence in raw_bindingdb_sequences:
        line = get_fasta_pair(sequence)
        if line is not None:
            mapped_bindingdb_sequences.append(line)

    with open("mapped_bindingdb_sequences.pkl", "wb") as f:
        pickle.dump(mapped_bindingdb_sequences, f)
else:
    mapped_bindingdb_sequences = pd.read_pickle("mapped_bindingdb_sequences.pkl")


### Pickle DUDE sequences.

# Download DUDE sequences.
if "mapped_dude_sequences.pkl" not in dir_files:
    def get_fasta(target):
        response = urlopen(f"https://www.rcsb.org/fasta/entry/{target.upper()}/display")
        return "\n" + response.read().decode("utf-8").strip("\n")

    mapped_dude_sequences = [
        get_fasta(target)
        for target in pd.read_csv("../DUDE/data_pre/dud-e_proteins.csv")["target_pdb"].tolist()
    ]
    mapped_dude_sequences = [sequence for sequence in mapped_dude_sequences
                                if "No fasta files were found." not in sequence]

    with open("mapped_dude_sequences.pkl", "wb") as f:
        pickle.dump(mapped_dude_sequences, f)
else:
    mapped_dude_sequences = pd.read_pickle("mapped_dude_sequences.pkl")

# Remove BindingDB sequences that are drugVQA sequences.
mapped_bindingdb_sequences = [line for line in mapped_bindingdb_sequences
                                if line.split("\n")[1] not in
                                mapped_dude_sequences]

# Write complete FASTA file.
with open("mapped_dude_sequences.txt", "w") as f:
    f.writelines(mapped_dude_sequences)
with open("mapped_bindingdb_sequences.txt", "w") as f:
    f.writelines(mapped_bindingdb_sequences)
with open("mapped_sequences_complete.txt", "w") as f:
    f.writelines(mapped_dude_sequences + mapped_bindingdb_sequences)

