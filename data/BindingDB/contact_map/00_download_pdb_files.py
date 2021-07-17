import os

import numpy as np
import pandas as pd

from Bio.PDB import *
from pickle import dump


def get_num_examples(sequence):
    return len([other_sequence
                for other_sequence in raw_sequences
                if other_sequence == sequence])


def get_sequence(sequence_list):
    sequence_num_examples = [get_num_examples(sequence) for sequence in sequence_list]
    if max(sequence_num_examples) < 40:
        return None
    return sequence_list[np.argmax(sequence_num_examples)]


# Read in sequences and PDB IDs.
bindingdb_examples = pd.read_csv("bindingdb_examples_raw.tsv", sep = "\t")

## Read in all sequences without dropping duplicates.
raw_sequences = [sequence.upper().strip().strip("\n").strip("\t")
                 for sequence in bindingdb_examples.target_sequence.dropna()]

## Create the sequence-PDB ID mapping.
bindingdb_examples = \
    bindingdb_examples[bindingdb_examples.columns
                        [bindingdb_examples.columns.isin(["target_sequence",
                                                          "target_pdb_id"])
                        ]
                      ].drop_duplicates().dropna()

raw_bindingdb_sequences = [sequence.upper().strip().strip("\n").strip("\t")
                           for sequence in bindingdb_examples.target_sequence]

sequence_to_id_map = dict(zip(raw_bindingdb_sequences,
                              [line.split(",")[0]
                               for line in bindingdb_examples.target_pdb_id.tolist()]))
sequence_to_id_map = {sequence: pdb_id for sequence, pdb_id
                      in sequence_to_id_map.items()
                      if len(sequence) != 7096 and pdb_id != "3MAX"}

selected_sequences = {pdb_id: [] for pdb_id in sequence_to_id_map.values()}
for sequence in sequence_to_id_map.keys():
    selected_sequences[sequence_to_id_map[sequence]].append(sequence)

for pdb_id in selected_sequences.keys():
    selected_sequences[pdb_id] = get_sequence(selected_sequences[pdb_id])

sequence_to_id_map = {sequence: pdb_id
                      for pdb_id, sequence in selected_sequences.items()
                      if sequence is not None}


# Download PDBs.
pdbl = PDBList()

pdbl.download_pdb_files(sequence_to_id_map.values(),
                        file_format="pdb", pdir="pdb_files")

with open("sequence_to_id_map.pkl", "wb") as f:
    dump(sequence_to_id_map, f)

# Write the processed bindingdb csv here.
bindingdb_examples = pd.read_csv("bindingdb_examples_raw.tsv", sep = "\t")
bindingdb_examples = bindingdb_examples[
                        bindingdb_examples.
                        target_sequence.isin(sequence_to_id_map.keys())
                     ]
bindingdb_examples.to_csv("../data_pre/bindingdb_examples.csv")
print(len(sequence_to_id_map))
print(len(bindingdb_examples))

