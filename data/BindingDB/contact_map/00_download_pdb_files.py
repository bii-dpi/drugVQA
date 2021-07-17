import os

import numpy as np
import pandas as pd

from Bio.PDB import *
from pickle import dump


def get_shortest(sequence_list):
    return sequence_list[np.argmin([len(sequence) for sequence in sequence_list])]


# Read in sequences and PDB IDs.
bindingdb_examples = pd.read_csv("bindingdb_examples_raw.tsv", sep = "\t")
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

selected_sequences = {pdb_id: [] for pdb_id in sequence_to_id_map.values()}
for sequence in sequence_to_id_map.keys():
    selected_sequences[sequence_to_id_map[sequence]].append(sequence)

for pdb_id in selected_sequences.keys():
    selected_sequences[pdb_id] = get_shortest(selected_sequences[pdb_id])

sequence_to_id_map = {sequence: pdb_id
                      for pdb_id, sequence in selected_sequences.items()}


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

