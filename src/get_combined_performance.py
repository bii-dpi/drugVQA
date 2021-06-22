import numpy as np
from utils import  *
from sklearn import metrics


SEEDS = [87459307486392109,
         48674128193724691,
         71947128564786214]


def get_AUPR(seed, fold_num, epoch):
    with open(f"../results/123456789_orig_rv_{seed}_rv_{fold_num}_validate_results.csv") as f:
        results = f.readlines()
    results = [line.split(",") for line in results]
    return [float(line[5]) for line in results if int(line[0]) == epoch][0]


def get_pred(seed, fold_num, epoch):
    return np.load(f"123456789_orig_rv_{seed}_rv_{fold_num}_bindingdb_{epoch}_pred.npy")


def get_target(seed, fold_num, epoch):
    return np.load(f"123456789_orig_rv_{seed}_rv_{fold_num}_bindingdb_{epoch}_target.npy")


def evaluate(seed, epoch):
    correct += torch.eq(torch.round(y_pred.squeeze(1)), y)
    all_pred = np.concatenate((all_pred, y_pred.data.cpu().squeeze(1).numpy()), axis = 0)
    all_target = np.concatenate((all_target, y.data.cpu().numpy()), axis = 0)

    loss = get_BCE(all_pred, all_target)
    accuracy = correct / len(all_target)
    recall = metrics.recall_score(all_target, np.round(all_pred))
    precision = metrics.precision_score(all_target, np.round(all_pred))
    AUC = metrics.roc_auc_score(all_target, all_pred)
    AUPR = metrics.average_precision_score(all_target, all_pred)

    roce_1 = get_ROCE(all_pred, all_target, 0.5)
    roce_2 = get_ROCE(all_pred, all_target, 1)
    roce_3 = get_ROCE(all_pred, all_target, 2)
    roce_4 = get_ROCE(all_pred, all_target, 5)

    with open(f"../results/combined_bindingdb_results_{seed}.csv", "a") as f:
        f.write((f"{epoch}, {accuracy}, {recall}, {precision}, {AUC}, {AUPR}, {loss}, "
                    f"{roce_1}, {roce_2}, {roce_3}, {roce_4}\n"))


for seed in SEEDS:
    for epoch in range(2, 51, 2):
        print(evaluate(seed, epoch))

