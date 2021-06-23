import numpy as np
from utils import  *
from sklearn import metrics


SEEDS = [87459307486392109,
         48674128193724691,
         71947128564786214]


def get_target(seed):
    return np.load(f"../model_pred/"
                   f"123456789_orig_rv_{seed}_rv_1_bindingdb_10_target.npy")

def test_equality(index_1, index_2):
    print(np.sum(get_target(SEEDS[index_1]) == \
                 get_target(SEEDS[index_2])) == \
          len(get_target(SEEDS[index_1])))

for i in range(0, 3):
    for j in range(0, 3):
        test_equality(i, j)

