import time
from contextlib import contextmanager

import numpy as np
from scipy import sparse
from sklearn.metrics import precision_recall_fscore_support


def _as_binary_matrix(y, name: str) -> np.ndarray:
    y = y.toarray() if sparse.issparse(y) else np.asarray(y)
    if y.ndim != 2:
        raise ValueError(f"{name} must be 2-D (issues x labels), got shape {y.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError(
            f"{name} must contain only 0/1. If these are model probabilities, "
            "apply a threshold first, e.g. (probs >= 0.5).astype(int)."
        )
    return y.astype(int)


def evaluate(y_true, y_pred) -> dict[str, float]:
    y_true = _as_binary_matrix(y_true, "y_true")
    y_pred = _as_binary_matrix(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def evaluate_by_frequency(y_true, y_pred, train_positives,
                          rare_cutoff: int = 10, common_cutoff: int = 100) -> dict:
    y_true = _as_binary_matrix(y_true, "y_true")
    y_pred = _as_binary_matrix(y_pred, "y_pred")
    counts = np.asarray(train_positives).ravel()
    groups = {
        "unseen": counts == 0,
        "rare": (counts > 0) & (counts < rare_cutoff),
        "medium": (counts >= rare_cutoff) & (counts < common_cutoff),
        "common": counts >= common_cutoff,
    }
    report = {}
    for name, cols in groups.items():
        if not cols.any():
            continue
        scores = evaluate(y_true[:, cols], y_pred[:, cols])
        report[name] = {
            "labels": int(cols.sum()),
            "labels_predicted": int((y_pred[:, cols].sum(axis=0) > 0).sum()),
            "support": int(y_true[:, cols].sum()),
            **{k: round(v, 4) for k, v in scores.items()},
        }
    return report


@contextmanager
def timer(label: str | None = None):
    record = {"seconds": None}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["seconds"] = round(time.perf_counter() - start, 3)
        if label:
            print(f"[{label}] {record['seconds']:.2f} s")