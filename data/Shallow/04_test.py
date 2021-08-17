import numpy as np
import autogluon.core as ag

from sklearn import metrics
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor
from autogluon.tabular import TabularPredictor, TabularDataset


SAMPLE_SIZE = 50000


def get_data(direction, resampling_ratio):
    print(f"Testing in direction {direction} with resampling ratio {resampling_ratio}.")
    print("Loading data...")
    return TabularDataset(f"{direction}_{resampling_ratio}_testing_examples.csv")[:SAMPLE_SIZE]


def save_performance(quartet):
    direction, resampling_ratio, random_seed, data = quartet

    predictor = TabularPredictor.load(f"models/{direction}_{resampling_ratio}_{random_seed}")

    for model in predictor.get_model_names():
        predictions = predictor.predict_proba(data, model)[1].tolist()

        recall = metrics.recall_score(data.Y, np.round(predictions))
        precision = metrics.precision_score(data.Y, np.round(predictions))
        AUPR = metrics.average_precision_score(data.Y, predictions)

        np.save(f"results/{direction}_{resampling_ratio}_{random_seed}_{model}.npy",
                [recall, precision, AUPR])


np.random.seed(12345)

random_seeds = np.random.randint(0, 999999999, 10)

for direction in progressbar(["btd", "dtb", "detb", "btde"]):
    data = get_data(direction, 1)
    quartets = []
    for random_seed in random_seeds:
        quartets.append([direction, 1, random_seed, data])
    with ProcessPoolExecutor() as executor:
        executor.map(save_performance, quartets)

