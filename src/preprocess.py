import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# T40 and T41 will completed in this file!
# Baseline settings (competition_baseline/main.py: TfidfVectorizer(max_features=1000)).
# T42 decides our final values; pass them as keyword arguments to build_tfidf().
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