def get_examples(i):
    with open(f"rv_{i}_val_fold", "r") as f:
        return f.readlines()


all_examples = []
for i in range(1, 4):
    all_examples += get_examples(i)

with open("shallow_training_examples", "w") as f:
    f.writelines(all_examples)

