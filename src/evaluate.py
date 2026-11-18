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