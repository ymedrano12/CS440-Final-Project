import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from db_config import load_competition_table
from preprocess import clean_text, build_tfidf
from sklearn.utils.multiclass import type_of_target


def train_sample():

    df = load_competition_table()
    df = df.sample(n=100, random_state=42).reset_index(drop=True)

    descriptions = df["issue description"].apply(clean_text).to_numpy()

    label_df = df.loc[:, "Application":]

    labels = (label_df > 0).astype(int).to_numpy()
   

    print("Label shape:", labels.shape)
    print("Label data type:", labels.dtype)
    print("Represented skills:", (labels.sum(axis=0) > 0).sum())
    print("Target type:", type_of_target(labels))
    label_df = df.loc[:, "Application":]

    print(label_df.iloc[:5, :10])

    print("Missing labels:", label_df.isna().sum().sum())

    print("Unique values per column:")
    print(label_df.nunique().value_counts())    

    
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42
    )

    
    vectorizer, X_train, X_test = build_tfidf(
        descriptions,
        train_idx,
        test_idx,
        max_features=1000,
        ngram_range=(1, 1)
    )

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
        n_jobs=1
    )

    
    print("Training model...")

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    end_time = time.perf_counter()

    print("Training complete!")
    print(f"Training time: {end_time - start_time:.2f} seconds")

    
    y_pred = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="micro",
        zero_division=0
    )

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


if __name__ == "__main__":
    train_sample()