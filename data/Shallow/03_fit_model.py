import numpy as np
import pandas as pd
import autogluon.core as ag
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.core.metrics import make_scorer
from sklearn.metrics import average_precision_score
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor


#scorer = make_scorer("AUPR", average_precision_score, optimum=1, needs_proba=True)

def get_data(direction, resample_ratio):
    return pd.read_csv(f"{direction}_{resample_ratio}_training_examples.csv")


def train(quartet):
    direction, resample_ratio, random_seed, data = quartet

    predictor = TabularPredictor(problem_type="binary", label="Y",
                                 eval_metric="f1",#scorer,
                                 path=f"models/{direction}_{resample_ratio}_{random_seed}")
    predictor.fit(train_data=data,
                  #time_limit=2.5*60*60,
                  presets="medium_quality_faster_train", #"good_quality_faster_inference_only_refit", #"medium_quality_faster_train"
                  hyperparameters={
                                     'KNN': {},
                                     'GBM': {"seed": random_seed},
                                     'RF': {"random_state": random_seed},
                                     'CAT': {"random_seed": random_seed},
                                     'XT': {"random_state": random_seed},
                                     'XGB': {"random_state": random_seed},
                                     'LR': {"random_state": random_seed}},
                  verbosity=0)


np.random.seed(12345)
random_seeds = np.random.randint(0, 999999999, 10)

print("Loading arguments...")
quartets = []
for direction in ["btde"]:# "btd", "dtb", "detb", "btde"]:
    data = get_data(direction, 1)
    for random_seed in random_seeds:
        quartets.append([direction, 1, random_seed, data])

print("Fitting model...")
with ProcessPoolExecutor() as executor:
    executor.map(train, quartets)
"""
for quartet in progressbar(quartets):
    train(quartet)
"""
