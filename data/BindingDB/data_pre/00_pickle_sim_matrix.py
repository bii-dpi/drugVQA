import pickle

import numpy as np

from urllib.request import urlopen


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
print(sim_matrix)

with open("sim_matrix.pkl", "wb") as f:
    pickle.dump(sim_matrix, f)

