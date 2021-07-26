def get_examples(i):
    with open(f"rv_{i}_val_fold", "r") as f:
        return [line.strip("\n") for line in f.readlines()]

all_examples = []
for i in range(1, 4):
    all_examples += get_examples(i)

with open("../../Shallow/dude_examples", "w") as f:
    f.write("\n".join(all_examples))

