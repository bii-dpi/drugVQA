import numpy as np

from progressbar import progressbar


np.random.seed(12345)

ILLEGAL_LIST = ["[c-]", "[N@@]", "[Re-]", "[S@@+]", "[S@+]"]


def has_illegal(line):
    smiles_string = line.split()[0]
    for illegal_element in ILLEGAL_LIST:
        if illegal_element in smiles_string:
            return True
    return False


def get_examples(protein_id):
    try:
        with open(f"results/{protein_id}_examples", "r") as f:
            examples = [line.strip().strip("\n") for line in f.readlines()]
        return [line for line in examples if not has_illegal(line)]
    except:
        return []


with open("data/all_actives", "r") as f:
    all_active_smiles = f.readlines()
all_protein_ids = np.unique([line.split()[1] for line in all_active_smiles])

all_examples = []
for protein_id in progressbar(all_protein_ids):
    all_examples += get_examples(protein_id)

"""
all_actives = [example for example in all_examples
               if int(example.split()[2])]
all_decoys = [example for example in all_examples
               if not int(example.split()[2])]
print(len(all_decoys) / len(all_actives))
np.random.shuffle(all_decoys)
all_decoys = all_decoys[:int(len(all_actives) * 50)]
all_examples = all_actives + all_decoys
"""
np.random.shuffle(all_examples)

print(f"Ratio: {np.mean([int(example.split()[2]) for example in all_examples])}")
print(f"Number of examples: {len(all_examples)}")

with open("results/bindingdb_test_examples", "w") as f:
    f.write("\n".join(all_examples))

with open("../drugVQA/data/BindingDB/data_pre/bindingdb_test_examples", "w") as f:
    f.write("\n".join(all_examples))

with open("../drugVQA/data/Shallow/shallow_testing_examples", "w") as f:
    f.write("\n".join(all_examples))

