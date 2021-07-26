import os

import numpy as np


def get_num_actives(fname):
    with open(fname, "r") as f:
        return len(f.readlines())


lens = [get_num_actives(fname) for fname in os.listdir()
        if fname.endswith(".ism")]


print(np.mean(lens), np.std(lens), np.max(lens), np.min(lens))
