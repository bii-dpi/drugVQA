from glob import glob


NUM_EPOCHS = 50
VAL_GAP = 2
FNAME_PREFIX = "orig_cv"


def get_epoch_dict(fname):
    with open(fname, "r") as f:
        epoch_list = [int(line.split(",")[0]) for line in f.readlines()]
    epoch_dict = {epoch: epoch_list.count(epoch)
                    for epoch in range(1, NUM_EPOCHS + 1)}
    return {epoch: count for epoch, count in epoch_dict.items()}


def check_train(fname):
    epoch_dict = get_epoch_dict(fname)
    zeroes = [epoch for epoch, count in epoch_dict.items()
                if not count]
    if zeroes:
        print(f"No {zeroes} in {fname}")

    dups = [(epoch, count) for epoch, count in epoch_dict.items()
            if count > 1]
    if dups:
        print(f"Duplicates {dups} in {fname}")


def check_val(fname):
    epoch_dict = get_epoch_dict(fname)
    epoch_dict = {epoch: count for epoch, count in epoch_dict.items()
                    if epoch in range(NUM_EPOCHS, 0, -VAL_GAP)}

    zeroes = [epoch for epoch, count in epoch_dict.items()
                if not count]
    if zeroes:
        print(f"No {zeroes} in {fname}")

    dups = [(epoch, count) for epoch, count in epoch_dict.items()
            if count > 1]
    if dups:
        print(f"Duplicates {dups} in {fname}")


all_train_files = glob(f"../results/{FNAME_PREFIX}*train_results.csv")
all_validate_files = glob(f"../results/{FNAME_PREFIX}*validate_results.csv")

for fname in all_train_files:
    check_train(fname)

for fname in all_validate_files:
    check_val(fname)

