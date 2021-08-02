import numpy as np
import autogluon.core as ag

from sklearn import metrics
from autogluon.tabular import TabularPredictor, TabularDataset


SAMPLE_SIZE = 40000


def print_performance(direction, resampling_ratio):
    print(f"Testing in direction {direction} with resampling ratio {resampling_ratio}.")
    print("Loading data...")
    data = TabularDataset(f"{direction}_{resampling_ratio}_testing_examples.csv")[:SAMPLE_SIZE]

    print("Testing model...")
    predictor = TabularPredictor.load(f"{direction}_{resampling_ratio}")
    predictions = predictor.predict_proba(data)[1].tolist()

    recall = metrics.recall_score(data.Y, np.round(predictions))
    precision = metrics.precision_score(data.Y, np.round(predictions))
    AUPR = metrics.average_precision_score(data.Y, predictions)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"AUPR: {AUPR}")


print_performance("btd", 1)
print_performance("dtb", 2)

