import pandas as pd


def get_ratio(target):
    with open(f"actives_smile/{target}_actives_final.ism", "r") as f:
        num_actives = len(f.readlines())

    with open(f"decoys_smile/{target}_decoys_final.ism", "r") as f:
        num_decoys = len(f.readlines())

    return num_decoys / num_actives


# Application
# Load all 102 target names.
all_targets = \
    pd.read_csv(f"dataPre/dud-e_proteins.csv")["name"]
all_targets = [target.lower() for target in all_targets]


for target in all_targets:
    print(get_ratio(target))
