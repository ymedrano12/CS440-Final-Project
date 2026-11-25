import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import type_of_target


def create_model(model_name, random_state=42, **params):

    if model_name == "random_forest":

        settings = {
            "n_estimators": 10,
            "random_state": random_state,
            "n_jobs": 1
        }

        settings.update(params)

        return RandomForestClassifier(**settings)

    elif model_name == "logistic_regression":

        settings = {
            "max_iter": 1000,
            "random_state": random_state,
            "C": 1.0
        }

        settings.update(params)

        return OneVsRestClassifier(
            LogisticRegression(**settings),
            n_jobs=1
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(model, X_train, y_train):

    # Validate label format
    if type_of_target(y_train) != "multilabel-indicator":
        raise ValueError("Labels must be multi-hot encoded")

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Feature and label dimensions do not match")

    print("Training shape:", X_train.shape)
    print("Label shape:", y_train.shape)
    print("Training model...")

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print("Training complete!")
    print(f"Training time: {training_time:.4f} seconds")

    return model, training_time


def evaluate_model(model, X_test, y_test):

    start_time = time.perf_counter()

    y_pred = model.predict(X_test)

    prediction_time = time.perf_counter() - start_time

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="micro",
        zero_division=0
    )

    results = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "prediction_time": prediction_time
    }

    print("Prediction shape:", y_pred.shape)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return results