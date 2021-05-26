import numpy as np
from pickle import dump
from copy import deepcopy
from scipy.stats import entropy
from progressbar import progressbar


INIT_SEED = 12345
NUM_TRIALS = int(1e2)


def get_simple_mix(seed):
    """Get mix using the simple method."""
    np.random.seed(seed)
    all_proteins = [protein for fold in ragoza_folds.values()
                    for protein in fold]
    np.random.shuffle(all_proteins)
    return {"CV1": all_proteins[:33],
            "CV2": all_proteins[33:69],
            "CV3": all_proteins[69:]}


def get_stratified_mix(seed):
    """Get mix using the stratified method."""
    np.random.seed(seed)
    ragoza_folds_copy = deepcopy(ragoza_folds)
    mixed_dict = {fold: [] for fold in ragoza_folds.keys()}
    for fold in ragoza_folds.keys():
        np.random.shuffle(ragoza_folds_copy[fold])
        for new_fold in mixed_dict.keys():
            for _ in range(len(ragoza_folds[fold]) // 3):
                mixed_dict[new_fold].append(ragoza_folds_copy[fold].pop())
    return mixed_dict


def get_fold_proportion(protein_list, fold):
    """Get proportion of each Ragoza fold in `protein_list`."""
    return np.mean(np.array([protein.split("_")[0]
                                for protein in protein_list]) == fold)

def get_entropy(protein_list):
    fold_props = [get_fold_proportion(protein_list, fold)
                    for fold in ["CV1", "CV2", "CV3"]]
    print(fold_props)
    return entropy(fold_props,
                    [33/102, 36/102, 33/102])


def get_mix_entropy(mixed_dict):
    """Get the entropy of the new folds in terms of Ragoza assignments."""
    # Get the average entropy of each new fold.
    print(np.mean([get_entropy(protein_list)
                    for protein_list in mixed_dict.values()]))
    return np.mean([get_entropy(protein_list)
                    for protein_list in mixed_dict.values()])


def get_seed(num_digits=9):
    return int("".join([str(np.random.randint(0, 9))
                        for _ in range(0, num_digits)]))


# Application
np.random.seed(INIT_SEED)

# Create "Ragoza folds".
ragoza_folds = {fold: [f"{fold}_{i}"
                            for i in range(1, 34)]
                for fold in ["CV1", "CV2", "CV3"]}
ragoza_folds["CV2"] += ["CV2_34", "CV2_35", "CV2_36"]


simple_mix_entropy = []
stratified_mix_entropy = []
for _ in progressbar(range(NUM_TRIALS)):
    curr_seed = get_seed()
    simple_mix_entropy.append(get_mix_entropy(get_simple_mix(curr_seed)))
    #stratified_mix_entropy.append(get_mix_entropy(get_stratified_mix(curr_seed)))


with open("simple_mix_entropy.pkl", "wb") as f:
    dump(simple_mix_entropy, f)
print(simple_mix_entropy)
print(np.mean(simple_mix_entropy))
print(np.std(simple_mix_entropy))
print(np.max(simple_mix_entropy))
print(np.min(simple_mix_entropy))

