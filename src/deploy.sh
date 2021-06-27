import os

FOLD_NUM = 1

for seed_index in range(0, 3):
    os.system(f"deploy.py train 3 {seed_index} rv {FOLD_NUM} {seed_index}")

