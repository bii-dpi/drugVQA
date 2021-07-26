import pandas as pd
import autogluon.core as ag
from autogluon.tabular import TabularPredictor, TabularDataset


def train(direction):
    print(f"Training in direction {direction}.")

    print("Loading data...")
    train_data = pd.read_csv(f"{direction}_training_examples.csv")


    print("Fitting model...")
    predictor = TabularPredictor(problem_type="binary", label="Y",
                                 eval_metric="f1", path=direction)
    predictor.fit(train_data=train_data,
                  time_limit=11*60*60,
presets="high_quality_fast_inference_only_refit",  #"best_quality", #"medium_quality_faster_train", #best_quality
                  excluded_model_types=["NN", "NeuralNetFastAI",
                                        "NNFastAiTabularModel",
                                        "TabularNeuralNetModel"], verbosity=2)
#train("btd")
train("dtb")

