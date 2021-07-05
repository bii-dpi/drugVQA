import numpy as np


with open("shallow_training_examples", "r") as f:
    training_examples = f.readlines()


np.random.seed(12345)
np.random.shuffle(training_examples)

validation_examples = training_examples[:int(0.2 * len(training_examples))]
training_examples = training_examples[int(0.2 * len(training_examples)):]


with open("shallow_training_examples", "w") as f:
    f.writelines(training_examples)

with open("shallow_validation_examples", "w") as f:
    f.writelines(validation_examples)

