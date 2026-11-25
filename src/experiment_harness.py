import time
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from db_config import load_competition_table
from preprocess import clean_text, build_tfidf

from models.random_forest import create_random_forest
from models.one_v_rest_logistic_regression import create_logistic_regression
from train import create_model, train_model, evaluate_model


def run_experiment(
    model_name,
    sample_size=100,
    max_features=1000,
    ngram_range=(1, 1),
    random_state=42
):

    
    df = load_competition_table()

    df = df.sample(
        n=sample_size,
        random_state=random_state
    ).reset_index(drop=True)

    
    descriptions = df["issue description"].apply(
        clean_text
    ).to_numpy()

    
    label_df = df.loc[:, "Application":]

    labels = (label_df > 0).astype(int).to_numpy()

    
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=random_state
    )

    
    vectorizer, X_train, X_test = build_tfidf(
        descriptions,
        train_idx,
        test_idx,
        max_features=max_features,
        ngram_range=ngram_range
    )

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    
    if model_name == "random_forest":
        model = create_random_forest(
            random_state=random_state
        )

    elif model_name == "logistic_regression":
        model = create_logistic_regression(
            random_state=random_state
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    
    print(f"\nTraining {model_name}...")

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    
    y_pred = model.predict(X_test)

    # 9. Evaluate
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="micro",
            zero_division=0
        )
    )

    
    results = {
        "model": model_name,
        "sample_size": sample_size,
        "training_samples": len(train_idx),
        "testing_samples": len(test_idx),
        "max_features": max_features,
        "actual_features": X_train.shape[1],
        "ngram_range": str(ngram_range),
        "random_state": random_state,
        "training_time": training_time,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print(f"Training time: {training_time:.4f} seconds")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return results

if __name__ == "__main__":

    results = []

    models = [
        #"random_forest",
        "logistic_regression"
    ]

    for model_name in models:

        result = run_experiment(
            model_name=model_name,
            sample_size=100,
            max_features=1000,
            ngram_range=(1, 1)
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    output_dir = Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        output_dir / "experiment_results.csv",
        index=False
    )

    print("\nExperiment Summary:")
    print(results_df.to_string(index=False))