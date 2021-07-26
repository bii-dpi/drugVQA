import numpy as np
import pandas as pd


def write_df(direction, mode):
    print(f"Writing {direction}_{mode}_examples.csv")
    features = np.load(f"{direction}_{mode}_examples_features_proc.npy")
    labels = np.expand_dims(np.load(f"{direction}_{mode}_examples_labels_proc.npy"), axis=1)
    data = np.concatenate((features, labels), axis=1)
    df = pd.DataFrame(data=data,
                      columns=[f"X{i}" for i in range(features.shape[1])] + \
                              ["Y"])
    df.to_csv(f"{direction}_{mode}_examples.csv", index=False)


for direction in ["dtb", "btd"]:
    for mode in ["training", "testing"]:
        write_df(direction, mode)

