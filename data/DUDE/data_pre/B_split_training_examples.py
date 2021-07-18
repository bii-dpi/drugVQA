import os
import numpy as np


with open("shallow_training_examples", "r") as f:
    training_examples = [line.strip("\n") for line in f.readlines()]


np.random.seed(12345)
np.random.shuffle(training_examples)

validation_examples = training_examples[:int(0.2 * len(training_examples))]
training_examples = training_examples[int(0.2 * len(training_examples)):]


with open("../../Shallow/shallow_training_examples", "w") as f:
    f.write("\n".join(training_examples))

with open("../../Shallow/shallow_validation_examples", "w") as f:
    f.write("\n".join(validation_examples))

os.system("rm -f shallow_training_examples")

