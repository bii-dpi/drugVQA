import numpy as np
import pandas as pd


def write_csv(path):
    features = np.load(f"{path}_features.npy")
    print(features.shape)
    labels = np.expand_dims(np.load(f"{path}_labels.npy"), axis=1)
    print(labels.shape)
    data = np.concatenate((features, labels), axis=1)
    df = pd.DataFrame(data=data,
                      columns=[f"X{i}" for i in range(features.shape[1])] + \
                              ["Y"])
    df.to_csv(f"{path}.csv", index=False)


example_paths = ["shallow_training_examples",
                 "shallow_testing_examples"]

[write_csv(path) for path in example_paths]


