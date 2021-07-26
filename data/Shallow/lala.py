import os
import numpy as np

with open("shallow_testing_examples", "r") as f:
    examples = [line.strip("\n") for line in f.readlines()]

sequences = np.unique([example.split()[1] for
                       example in examples])

sequences = [sequence for sequence in sequences
             if len(sequence) <= 400]

examples = [example for example in examples
            if example.split()[1] in sequences]

print(len(examples))
print(examples[:10])

with open("../BindingDB/data_pre/bindingdb_examples", "w") as f:
    f.write("\n".join(examples))
