import numpy as np


SEED_INDEX = 3

SEED = [123456789,
        619234965,
        862954379,
        296493420,
        579340210]

np.random.seed(SEED[SEED_INDEX])

def get_ragoza_fold_proteins(fold):
    with open(f"orig_val_{fold}", "r") as f:
        protein_list = [protein_name.split("_")[0] for protein_name
                        in f.read().split()]
    np.random.shuffle(protein_list)
    return protein_list


def write_new_fold(fold):
    validation_proteins = []
    for ragoza_fold in range(1, 4):
        curr_third_len = len(ragoza_fold_proteins[ragoza_fold]) // 3
        validation_proteins += \
            ragoza_fold_proteins[ragoza_fold][(fold - 1) * curr_third_len: fold * curr_third_len]
    training_proteins = [protein for protein in all_proteins
                            if protein not in validation_proteins]

    with open(f"rv_val_proteins_{fold}", "w") as f:
        f.write(" ".join(validation_proteins))

    with open(f"rv_train_proteins_{fold}", "w") as f:
        f.write(" ".join(training_proteins))

    return validation_proteins


def get_ragoza_fold_id(protein):
    for fold in range(1, 4):
        if protein in ragoza_fold_proteins[fold]:
            return fold
    raise Exception(f"{protein} does not belong to any fold.")


def get_cv_fold_props(protein_list):
    ragoza_fold_ids = np.array([get_ragoza_fold_id(protein)
                                for protein in protein_list])
    return [np.mean(ragoza_fold_ids == fold)
            for fold in range(1, 4)]


ragoza_fold_proteins = {fold: get_ragoza_fold_proteins(fold)
                        for fold in range(1, 4)}
all_proteins = [protein for protein_list in ragoza_fold_proteins.values()
                for protein in protein_list]

random_fold_proteins = {}
for fold in range(1, 4):
    random_fold_proteins[fold] = write_new_fold(fold)

assert len(set([protein for fold in range(1, 4)
                for protein in random_fold_proteins[fold]])) == 102

print([get_cv_fold_props(random_fold_proteins[fold]) for fold in range(1, 4)])
print([33/102, 36/102, 33/102])


