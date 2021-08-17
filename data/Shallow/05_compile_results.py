import os

import numpy as np
import pandas as pd


def get_entry(direction, resampling_ratio, model, all_seeds, index):
    values = []
    for seed in all_seeds:
        values.append(np.load(f"results/{direction}_{resampling_ratio}_{seed}_{model}.npy")[index])
    return f"{np.mean(values):.3f} $\pm$ {np.std(values):.3f}"


def save_performance(direction, resampling_ratio):
    fnames = [fname for fname in os.listdir("results/")
              if fname.startswith(f"{direction}_{resampling_ratio}_")]
    all_models = np.unique([fname.split(".npy")[0].split("_")[3]
                            for fname in fnames])
    all_models = [model for model in all_models
                  if model != "WeightedEnsemble"] + ["WeightedEnsemble_L2"]
    all_seeds = np.unique([fname.split(".npy")[0].split("_")[2]
                           for fname in fnames])

    results = {"model": [], "recall": [], "precision": [], "AUPR": []}
    for model in all_models:
        results["model"].append(model)
        results["recall"].append(get_entry(direction, resampling_ratio, model, all_seeds, 0))
        results["precision"].append(get_entry(direction, resampling_ratio, model, all_seeds, 1))
        results["AUPR"].append(get_entry(direction, resampling_ratio, model, all_seeds, 2))

    pd.DataFrame.from_dict(results).to_csv(f"results/overall_{direction}_{resampling_ratio}_results.csv",
                                           index=False)


for direction in ["btd", "dtb", "detb", "btde"]:
    save_performance(direction, 1)

