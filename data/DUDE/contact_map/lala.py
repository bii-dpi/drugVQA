import numpy as np

with open("DUDE_contactdict", "r") as f:
    text = f.readlines()

sequences = [len(line.split(":")[0]) for line in text]
print(np.unique(sequences))

