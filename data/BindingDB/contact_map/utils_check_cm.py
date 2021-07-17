import os

import numpy as np
import pandas as pd


sequence_to_id = pd.read_pickle("sequence_to_id_map.pkl")

print(len(sequence_to_id) -
      np.sum([1 for sequence in sequence_to_id.keys()
              if sequence_to_id[sequence] + "_cm" in os.listdir()]))

