from __future__ import annotations

import re
import string

import emoji
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from db_config import load_competition_table

# TF-IDF settings (T42): same as the baseline.
# Larger vocabularies (5000, 10000) and bigrams (1, 2) are left for the
# hyperparameter search; pass them as keyword arguments to build_tfidf().
DEFAULT_TFIDF_SETTINGS = {"max_features": 1000, "ngram_range": (1, 1)}


def build_tfidf(texts, train_idx, *other_idx, **settings):

    texts = np.asarray(texts, dtype=object)
    train_idx = np.asarray(train_idx)
    for idx in other_idx:
        overlap = np.intersect1d(train_idx, idx)
        if overlap.size:
            raise ValueError(
                f"{overlap.size} rows are in both the training split and another "
                "split; TF-IDF would be fit on evaluation rows."
            )

    vectorizer = TfidfVectorizer(**{**DEFAULT_TFIDF_SETTINGS, **settings})
    X_train = vectorizer.fit_transform(texts[train_idx])
    X_others = [vectorizer.transform(texts[np.asarray(idx)]) for idx in other_idx]
    return (vectorizer, X_train, *X_others)

def clean_text(text):
    text = handle_null_text(text)

    text = text.replace('"', "")
    text = re.sub(r"DevTools.*?\(automated\)", "", text)
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) #replace w/ a space instead?
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    words = [word for word in words if len(word) <= 20] #check diff restrictions?
    cleaned_text = " ".join(words)

    return cleaned_text

def handle_null_text(text):
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        return str(text)

    return text

def test_clean_text():
    df = load_competition_table()

    sample = df.sample(n=5, random_state=42).copy()

    sample["cleaned_issue_text"] = (
        sample["issue text"].apply(clean_text)
    )

    sample["cleaned_description"] = (
        sample["issue description"].apply(clean_text)
    )

    for _, row in sample.iterrows():
        print("Original issue:", row["issue text"])
        print("Cleaned issue:", row["cleaned_issue_text"])

        print("Original description:", row["issue description"])
        print("Cleaned description:", row["cleaned_description"])

        print("-" * 80)

def test_actual_tfidf():
    df = load_competition_table()

    sample = df.sample(n=5, random_state=42).copy()

    train_idx = np.array([0, 1, 2])
    test_idx = np.array([3, 4])

    for column in ["issue text", "issue description"]:

        texts = sample[column].apply(clean_text).to_numpy()

        print("\nTesting column:", column)

        for ngram in [(1, 1), (2, 2), (1, 2)]:

            vectorizer, X_train, X_test = build_tfidf(
                texts,
                train_idx,
                test_idx,
                max_features=1000,
                ngram_range=ngram
            )

            print("\nN-gram range:", ngram)
            print("Vocabulary size:", len(vectorizer.vocabulary_))
            print("Features:", vectorizer.get_feature_names_out()[:20])

            print("Training shape:", X_train.shape)
            print("Testing shape:", X_test.shape)

            print("Training nonzero values:", X_train.nnz)
            print("Testing nonzero values:", X_test.nnz)

            print("-" * 50)




if __name__ == "__main__":
    test_actual_tfidf()
    test_clean_text()