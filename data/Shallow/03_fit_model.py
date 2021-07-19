import autogluon.core as ag
from autogluon.tabular import TabularPredictor, TabularDataset

print("Loading data...")
train_data = TabularDataset("shallow_training_examples.csv")

print("Fitting model...")
predictor = TabularPredictor(problem_type="binary", label="Y",
                             eval_metric="f1", path="ag_models")
predictor.fit(train_data=train_data,
              time_limit=11*60*60, presets="medium_quality_faster_train",
              excluded_model_types=["NN", "NeuralNetFastAI"])

