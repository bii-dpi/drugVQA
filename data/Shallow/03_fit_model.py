import pandas as pd
import autogluon.core as ag
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.core.metrics import make_scorer
from sklearn.metrics import average_precision_score


scorer = make_scorer("AUPR", average_precision_score, optimum=1, needs_proba=True)

def train(direction, resample_ratio):
    print(f"Training in direction {direction} with ratio {resample_ratio}.")

    print("Loading data...")
    train_data = pd.read_csv(f"{direction}_{resample_ratio}_training_examples.csv")


    print("Fitting model...")
    predictor = TabularPredictor(problem_type="binary", label="Y",
                                 eval_metric=scorer,
                                 path=f"{direction}_{resample_ratio}")
    predictor.fit(train_data=train_data,
                  time_limit=11*60*60,
                  presets="good_quality_faster_inference_only_refit",
                  excluded_model_types=["NN", "NeuralNetFastAI",
                                        "NNFastAiTabularModel",
                                        "TabularNeuralNetModel"],
                  verbosity=2)

train("btd", 1)
#train("dtb", 2)

