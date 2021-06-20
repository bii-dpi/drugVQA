import numpy as np
import pandas as pd

from pickle import dump


bindingdb_examples = pd.read_csv("bindingdb_examples.tsv", sep = "\t")
bindingdb_examples = \
    bindingdb_examples[bindingdb_examples.columns
                        [bindingdb_examples.columns.isin(["target_sequence",
                                                          "target_pdb_id"])
                        ]
                      ].drop_duplicates().dropna()

raw_bindingdb_sequences = [sequence.upper().strip().strip("\n").strip("\t") for sequence
                            in bindingdb_examples.target_sequence]
print(len(raw_bindingdb_sequences))
with open("raw_bindingdb_sequences.pkl", "wb") as f:
    dump(raw_bindingdb_sequences, f)

with open("../contact_map/sequence_to_id_map.pkl", "wb") as f:
    dump(dict(zip(raw_bindingdb_sequences,
                  bindingdb_examples.target_pdb_id.tolist())),
         f)

