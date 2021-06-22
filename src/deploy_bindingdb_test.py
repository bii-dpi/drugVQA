import os

for seed_index in range(0, 3):
	for fold_num in range(1, 4):
		print(seed_index, fold_num)
		# RV_SEED_INDEX, SEED_INDEX, FOLD_TYPE, FOLD_NUM, CUDA_NUM
		os.system(f"python test_bindingdb_single.py 0 {seed_index} rv {fold_num} 0")

