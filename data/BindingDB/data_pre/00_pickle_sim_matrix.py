import os
import pickle

import numpy as np
import pandas as pd

from urllib.request import urlopen


def get_sequence(fname):
    with open(fname, "r") as f:
        return f.readlines()[1].strip("\n")


# There are many duplicate PDB IDs, but there are far fewer duplicate FASTA
# headings, even for the same PDB ID. Since my testing set needs to be diverse,
# here, I will keep only one from each PDB ID. For simplicity, I will keep the
# sequences that happen to be in the _cm files.
print("Loading selections...")
selected_sequences = [get_sequence(f"../contact_map/{fname}")
                      for fname in os.listdir("../contact_map")
                      if fname.endswith("_cm")]
pdb_ids = [fname.split("_")[0] for fname in os.listdir("../contact_map")
           if fname.endswith("_cm")]

with open("../contact_map/all_fastas_proc", "r") as f:
    all_fastas = f.readlines()
fasta_dict = dict(zip([all_fastas[i] for i in range(1, len(all_fastas), 2)],
                      [all_fastas[i] for i in range(0, len(all_fastas), 2)]))
selected_headers = [fasta_dict[sequence] for sequence in selected_sequences]

print(selected_headers[:2])




















"""
with open("sim_url", "r") as f:
    sim_url = f.read()

sim_matrix = urlopen(sim_url).read().decode('utf-8')
sim_matrix = [line.split()[1:] for line in sim_matrix.split("\n")[6:-1]]

# The headings in the matrix are all unique in their full form, since they are
# derived from all_fastas_proc. I will preserve them here so that I can know
# which entries to remove before I convert the keys to just their PDB form.
sim_matrix = {line[0]: np.array(line[1:], dtype=float)
              for line in sim_matrix}
sim_matrix = {key: dict(zip(sim_matrix.keys(), value))
              for key, value in sim_matrix.items()}

unique_pdb_ids = np.unique([key.split("_")[0]
                            for key in sim_matrix.keys()]).tolist()
heading_assignment = {pdb_id: None for pdb_id in unique_pdb_ids}

for heading in sim_matrix.keys():
    corr_pdb_id = heading.split("_")[0]
    for pdb_id in unique_pdb_ids:
        if heading_assignment[pdb_id]:
            continue
        if pdb_id == corr_pdb_id:
            heading_assignment[pdb_id] = heading

sim_matrix = {heading: value for heading, value in sim_matrix.items()
              if heading in heading_assignment.values()}
for heading in sim_matrix.keys():
    sim_matrix[heading] = {subheading: sim for subheading, sim
                           in sim_matrix[heading].items()
                           if subheading in heading_assignment.values()}

# The last step will be to make things in terms of sequences only.


with open("../contact_map/all_fastas_proc", "r") as f:
    text = f.readlines()
fasta_dict = dict(zip([text[i].strip("\n") for i in range(0, len(text), 2)],
                      [text[i].strip("\n") for i in range(1, len(text), 2)]))

heading_to_sequence = {}
for heading in sim_matrix.keys():
    for full_heading in fasta_dict.keys():
        if heading in full_heading:
            heading_to_sequence[heading] = fasta_dict[full_heading]

sim_matrix = {heading_to_sequence[heading]:
              {heading_to_sequence[subheading]: sim
               for subheading, sim in value.items()}
              for heading, value in sim_matrix.items()}

with open("sim_matrix.pkl", "wb") as f:
    pickle.dump(sim_matrix, f)

bindingdb_examples = pd.read_csv("../contact_map/bindingdb_examples_raw.tsv", sep = "\t")

"target_sequence"
# Save filtered examples
#bindingdb_examples.to_csv("bindingdb_examples_proc.csv")
"""
