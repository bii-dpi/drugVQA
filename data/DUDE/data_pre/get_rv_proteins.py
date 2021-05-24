import numpy as np


np.random.seed(123456789)


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



ragoza_fold_proteins = {fold: get_ragoza_fold_proteins(fold)
                        for fold in range(1, 4)}
all_proteins = [protein for protein_list in ragoza_fold_proteins.values()
                for protein in protein_list]

for fold in range(1, 4):
    write_new_fold(fold)

