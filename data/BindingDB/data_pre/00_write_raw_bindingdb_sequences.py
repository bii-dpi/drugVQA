import os

import numpy as np
import pandas as pd

from pickle import dump


def has_cm(pdb_ids):
    return f"{pdb_ids.split(',')[0].upper()}_cm" in os.listdir("../contact_map")


bindingdb_examples = pd.read_csv("bindingdb_examples.tsv", sep = "\t")
bindingdb_examples = \
    bindingdb_examples[bindingdb_examples.columns
                        [bindingdb_examples.columns.isin(["target_sequence",
                                                          "target_pdb_id"])
                        ]
                      ].drop_duplicates().dropna()


raw_bindingdb_sequences = [sequence.upper().strip().strip("\n").strip("\t")
                           for sequence in bindingdb_examples.target_sequence]

sequence_to_id_map = dict(zip(raw_bindingdb_sequences,
                              bindingdb_examples.target_pdb_id.tolist()))

sequence_to_id_map = {sequence:pdb_ids for sequence, pdb_ids in sequence_to_id_map.items()
                        if has_cm(pdb_ids)}

with open("raw_bindingdb_sequences.pkl", "wb") as f:
    dump(list(sequence_to_id_map.keys()), f)

with open("../contact_map/sequence_to_id_map.pkl", "wb") as f:
    dump(sequence_to_id_map, f)

