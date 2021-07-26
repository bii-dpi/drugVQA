import numpy as np
import autogluon.core as ag

from sklearn import metrics
from autogluon.tabular import TabularPredictor, TabularDataset


SAMPLE_SIZE = 40000


def print_performance(direction):
    print(f"Testing in direction {direction}.")
    print("Loading data...")
    data = TabularDataset(f"{direction}_testing_examples.csv")[:SAMPLE_SIZE]

    print("Testing model...")
    predictor = TabularPredictor.load(direction)
    predictions = predictor.predict_proba(data)[1].tolist()

    recall = metrics.recall_score(data.Y, np.round(predictions))
    precision = metrics.precision_score(data.Y, np.round(predictions))
    AUPR = metrics.average_precision_score(data.Y, predictions)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"AUPR: {AUPR}")


print_performance("dtb")
print_performance("btd")

