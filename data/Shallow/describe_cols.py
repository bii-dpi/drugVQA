import numpy as np
import pandas as pd

def print_desc(mode):
    data = pd.read_csv(f"shallow_{mode}_examples.csv")

    desc = data.describe()

    means = [round(mean, 2) for mean in desc.loc["mean", :].tolist()]
    print([i for i in range(len(means))
           if np.isnan(means[i])])
    print([i for i in range(len(means))
           if np.isinf(means[i])])

print_desc("training")
print_desc("testing")
