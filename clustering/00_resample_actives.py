import numpy as np


with open("data/all_actives", "r") as f:
    all_actives = [line.split() for line in f.readlines()]


all_pdb_ids = np.unique([line[1] for line in all_actives])
pdb_counts = dict(zip(all_pdb_ids,
                      [0 for pdb_id in all_pdb_ids]))

for pdb_id in all_pdb_ids:
    pdb_counts[pdb_id] = len([line for line in all_actives
                              if line[1] == pdb_id])

pdb_counts = list(pdb_counts.values())

print(np.mean(pdb_counts), np.std(pdb_counts),
      np.max(pdb_counts), np.min(pdb_counts))
