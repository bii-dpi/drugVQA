import os

import numpy as np
import pandas as pd

from Bio.PDB import *
from pickle import dump


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

# Download PDBs.
pdbl = PDBList()

pdbl.download_pdb_files(sequence_to_id_map.values(),
                        file_format="pdb", pdir="pdb_files")

with open("sequence_to_id_map.pkl", "wb") as f:
    dump(sequence_to_id_map, f)

"""
with open("raw_bindingdb_sequences.pkl", "wb") as f:
    dump(list(sequence_to_id_map.keys()), f)
"""

