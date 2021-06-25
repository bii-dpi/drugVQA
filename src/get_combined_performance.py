import numpy as np
import pandas as pd
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
    return np.load(f"../model_pred/"
                   f"123456789_orig_rv_{seed}_rv_{fold_num}_bindingdb_{epoch}_pred.npy")


def get_target(seed, fold_num, epoch):
    return np.load(f"../model_pred/"
                   f"123456789_orig_rv_{seed}_rv_{fold_num}_bindingdb_{epoch}_target.npy")


def combine_pred(epoch, method):
    if method == "mean":
        total_pred = None
        for seed in SEEDS:
            for fold_num in range(1, 4):
                if total_pred is None:
                    total_pred = get_pred(seed, fold_num, epoch)
                else:
                    total_pred += get_pred(seed, fold_num, epoch)
        return total_pred / (len(SEEDS) * len(range(1, 4)))
    elif method == "weighted":
        weights_dict = {}
        for seed in SEEDS:
            for fold_num in range(1, 4):
                weights_dict[f"{seed}_{fold_num}"] = get_AUPR(seed, fold_num, epoch)
        normalized_weights = np.array(list(weights_dict.values()))
        normalized_weights /= normalized_weights.sum()
        weights_dict = dict(zip(weights_dict.keys(),
                                normalized_weights.tolist()))

        pred = None
        for seed in SEEDS:
            for fold_num in range(1, 4):
                if pred is None:
                    pred = weights_dict[f"{seed}_{fold_num}"] * get_pred(seed, fold_num, epoch)
                else:
                    pred += weights_dict[f"{seed}_{fold_num}"] * get_pred(seed, fold_num, epoch)
        return pred
    elif method == "mean_excluded":
        total_pred = None
        for seed in SEEDS:
            for fold_num in range(1, 4):
                if seed == 48674128193724691 and fold_num in [1, 2]:
                    continue
                if total_pred is None:
                    total_pred = get_pred(seed, fold_num, epoch)
                else:
                    total_pred += get_pred(seed, fold_num, epoch)
        return total_pred / (len(SEEDS) * len(range(1, 4)) - 2)
    elif method == "weighted_excluded":
        weights_dict = {}
        for seed in SEEDS:
            for fold_num in range(1, 4):
                if seed == 48674128193724691 and fold_num in [1, 2]:
                    continue
                weights_dict[f"{seed}_{fold_num}"] = get_AUPR(seed, fold_num, epoch)
        normalized_weights = np.array(list(weights_dict.values()))
        normalized_weights /= normalized_weights.sum()
        weights_dict = dict(zip(weights_dict.keys(),
                                normalized_weights.tolist()))

        pred = None
        for seed in SEEDS:
            for fold_num in range(1, 4):
                if seed == 48674128193724691 and fold_num in [1, 2]:
                    continue
                if pred is None:
                    pred = weights_dict[f"{seed}_{fold_num}"] * get_pred(seed, fold_num, epoch)
                else:
                    pred += weights_dict[f"{seed}_{fold_num}"] * get_pred(seed, fold_num, epoch)
        return pred
    else:
        raise Exception(f"Method {method} not allowed.")


def evaluate(epoch, method):
    all_target = get_target(SEEDS[0], 1, epoch)
    all_pred = combine_pred(epoch, method)

    loss = metrics.log_loss(all_target, all_pred)
    accuracy = np.mean(all_target == all_pred)
    recall = metrics.recall_score(all_target, np.round(all_pred))
    precision = metrics.precision_score(all_target, np.round(all_pred))
    AUC = metrics.roc_auc_score(all_target, all_pred)
    AUPR = metrics.average_precision_score(all_target, all_pred)
    print(f"Method {method}: {np.mean(all_pred):.2f}, {AUPR:.2f}")

    roce_1 = get_ROCE(all_pred, all_target, 0.5)
    roce_2 = get_ROCE(all_pred, all_target, 1)
    roce_3 = get_ROCE(all_pred, all_target, 2)
    roce_4 = get_ROCE(all_pred, all_target, 5)

    with open(f"../results/combined_bindingdb_results_{method}.csv", "a") as f:
        f.write((f"{epoch}, {accuracy}, {recall}, {precision}, {AUC}, {AUPR}, {loss}, "
                    f"{roce_1}, {roce_2}, {roce_3}, {roce_4}\n"))



os.system(f"rm ../results/combined_bindingdb_results_mean.csv")
os.system(f"rm ../results/combined_bindingdb_results_weighted.csv")

for epoch in range(2, 51, 2):
    print(f"Epoch {epoch}")
    evaluate(epoch, "mean")
    evaluate(epoch, "weighted")
    evaluate(epoch, "mean_excluded")
    evaluate(epoch, "weighted_excluded")
    print("----------------")


def get_corr_matrix(epoch):
    preds_matrix = {}
    for seed in SEEDS:
        for fold_num in range(1, 4):
            preds_matrix[f"{seed}_{fold_num}"] = get_pred(seed, fold_num, epoch)
    return pd.DataFrame.from_dict(preds_matrix).corr()


corr_matrix = None
for epoch in range(2, 51, 2):
    if corr_matrix is None:
        corr_matrix = get_corr_matrix(epoch)
    else:
        corr_matrix += get_corr_matrix(epoch)
corr_matrix /= len(range(2, 51, 2))

corr_matrix.to_csv("corr_matrix.csv")
