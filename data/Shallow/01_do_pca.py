import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


RESAMPLING_RATIO = 1
np.random.seed(12345)


def get_resampled_data(features, labels):
    positive_label_indices = np.where(labels == 1)[0]
    positive_features = features[positive_label_indices, :]

    negative_label_indices = np.where(labels == 0)[0]
    np.random.shuffle(negative_label_indices)
    negative_label_indices = negative_label_indices[:(len(positive_label_indices)
                                                      * RESAMPLING_RATIO)]
    negative_features = features[negative_label_indices, :]

    features = np.concatenate([positive_features, negative_features])
    labels = np.array([1 for index in positive_label_indices] +
                      [0 for index in negative_label_indices])
    shuffling_indices = list(range(len(labels)))
    np.random.shuffle(shuffling_indices)
    features = features[shuffling_indices, :]
    labels = labels[shuffling_indices]

    return features, labels


def is_safe(col):
    col_sum = np.sum(col)
    return not (np.isnan(col_sum) or np.isinf(col_sum))


def save_processed_data(direction):
    features_base = "examples_features.npy"
    labels_base = "examples_labels.npy"

    if direction == "dtb":
        training = "dude"
        testing = "bindingdb"
    elif direction == "btd":
        training = "bindingdb"
        testing = "dude"

    print("")
    print(f"Processing for training: {training}, testing: {testing}")

    print(f"Resampling training data with ratio 1:{RESAMPLING_RATIO}...")
    training_features, training_labels = \
        get_resampled_data(np.load(f"{training}_{features_base}"),
                           np.load(f"{training}_{labels_base}"))

    testing_features, testing_labels = \
        np.load(f"{testing}_{features_base}"), \
        np.load(f"{testing}_{labels_base}")

    print("Checking which columns are safe...")
    safe_columns_training = np.apply_along_axis(is_safe,
                                                axis=0,
                                                arr=training_features)
    safe_columns_testing = np.apply_along_axis(is_safe,
                                               axis=0,
                                               arr=testing_features)
    safe_columns = [safe_columns_training[i] and safe_columns_testing[i]
                    for i in range(len(safe_columns_training))]
    print(f"Removed {training_features.shape[1] - np.sum(safe_columns)} columns.")
    training_features = training_features[:, safe_columns]
    testing_features = testing_features[:, safe_columns]

    print("Scaling data...")
    scaler = RobustScaler()
    training_features = scaler.fit_transform(training_features)
    testing_features = scaler.transform(testing_features)

    print("Doing PCA...")
    pca = PCA(n_components=0.95, whiten=True, random_state=12345)
    training_features = pca.fit_transform(training_features)
    testing_features = pca.transform(testing_features)
    print(f"Selecting {training_features.shape[1]} components.")

    print("Saving processed features...")
    np.save(f"{direction}_training_examples_features_proc.npy",
            training_features)
    np.save(f"{direction}_training_examples_labels_proc.npy",
            training_labels)
    np.save(f"{direction}_testing_examples_features_proc.npy",
            testing_features)
    np.save(f"{direction}_testing_examples_labels_proc.npy",
            testing_labels)

    print("Done.")


save_processed_data("btd")
save_processed_data("dtb")

