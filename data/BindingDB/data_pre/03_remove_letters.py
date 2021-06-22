import os
import pickle

import numpy as np
import pandas as pd

from urllib.request import urlopen


NUM_SELECTED = 25


with open(f"bindingdb_examples_filtered_{NUM_SELECTED}", "w") as f:
    text = f.write("\n".join(filtered_examples))

print(text)
