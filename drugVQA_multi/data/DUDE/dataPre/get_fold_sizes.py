import numpy as np
import pandas as pd

def print_numbers(path):
    with open(path, "r") as f:
        examples = f.readlines()
    examples = [int(example.split()[-1]) for example in examples]
    num_actives = np.sum(np.array(examples) == 1)
    print(path, num_actives, len(examples) - num_actives,
            (len(examples) - num_actives) / num_actives) 

paths = [f"cv_train_{i}" for i in range(1, 4)] + \
            [f"cv_val_{i}" for i in range(1, 4)] + \
            ["single_training_fold", "single_validation_fold"]

[print_numbers(path) for path in paths]

