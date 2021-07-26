import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs


def get_sims(pdb_id):
    sims = []
    for subdict in actives_dict[pdb_id]:
        for other_subdict in actives_dict[pdb_id]:
            if list(subdict.keys())[0] == list(other_subdict.keys())[0]:
                continue
            sims.append(DataStructs.FingerprintSimilarity(list(subdict.values())[0],
                                                          list(other_subdict.values())[0]))
    return sims


with open("../../../clustering/data/all_actives", "r") as f:
    actives = [line.split() for line in f.readlines()]

actives_dict = {}
for example in actives:
    try:
        actives_dict[example[1]].append({example[0]:
                                         Chem.RDKFingerprint(Chem.MolFromSmiles(example[0]))})
    except:
        actives_dict[example[1]] = [{example[0]:
                                     Chem.RDKFingerprint(Chem.MolFromSmiles(example[0]))}]

data = {"pdb_id": [],
        "mean_sim": [], "std_sim": [],
        "max_sim": [], "min_sim": []}
for pdb_id in actives_dict.keys():
    data["pdb_id"].append(pdb_id)
    sims = get_sims(pdb_id)
    print(np.mean([sim >= 0.8 for sim in sims]))
    data["mean_sim"].append(np.mean(sims))
    data["std_sim"].append(np.std(sims))
    data["max_sim"].append(np.max(sims))
    data["min_sim"].append(np.min(sims))

pd.DataFrame.from_dict(data).to_csv("sim_stats.csv")

