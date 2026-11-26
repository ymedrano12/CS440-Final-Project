import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from db_config import REPO_ROOT, load_competition_table
from evaluate import evaluate, evaluate_by_frequency, timer
from make_split import (N_METADATA_COLUMNS, TEXT_COLUMNS, load_split,
                        split_validation)
from mlp import (DEFAULT_POS_WEIGHT_CAP, build_loss, build_mlp,
                 compute_pos_weight)
from preprocess import build_tfidf, clean_text
from train_mlp import predict_proba, train_mlp

RESULTS_DIR = REPO_ROOT / "results" / "mlp"


def issue_texts(df: pd.DataFrame) -> np.ndarray:
    title = df[TEXT_COLUMNS[0]].map(clean_text)
    body = df[TEXT_COLUMNS[1]].map(clean_text)
    return (title + " " + body).str.strip().to_numpy()


def label_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return (df.iloc[:, N_METADATA_COLUMNS:].fillna(0) > 0).astype(np.int8)


def check_one_hot(y_pred: np.ndarray, n_rows: int, n_labels: int) -> dict:
    if y_pred.shape != (n_rows, n_labels):
        raise ValueError(f"Expected shape {(n_rows, n_labels)}, got {y_pred.shape}")
    values = set(np.unique(y_pred).tolist())
    if not values <= {0, 1}:
        raise ValueError(f"Prediction matrix contains values other than 0/1: {values}")
    per_issue = y_pred.sum(axis=1)
    return {
        "shape": list(y_pred.shape),
        "values": sorted(values),
        "mean_labels_per_issue": round(float(per_issue.mean()), 3),
        "issues_with_no_label": int((per_issue == 0).sum()),
        "labels_never_predicted": int((y_pred.sum(axis=0) == 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="optional path to the database")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability cut-off for a label (tuned on validation later)")
    ap.add_argument("--weighted", action="store_true",
                    help="weight each label's positives by negatives/positives (training rows only)")
    ap.add_argument("--pos-weight-cap", type=float, default=DEFAULT_POS_WEIGHT_CAP,
                    help="largest per-label weight used with --weighted")
    args = ap.parse_args()

    df = load_competition_table(args.db)
    texts, y = issue_texts(df), label_matrix(df)

    train_idx, _test_idx = load_split(df, args.seed)  # test stays unused until T63
    fit_idx, val_idx = split_validation(df, train_idx, seed=args.seed)
    print(f"Rows: fit {len(fit_idx)} | validation {len(val_idx)} | "
          f"test {len(_test_idx)} (held out)")

    _, X_fit, X_val = build_tfidf(texts, fit_idx, val_idx)
    y_fit, y_val = y.iloc[fit_idx], y.iloc[val_idx]
    print(f"TF-IDF features: {X_fit.shape[1]} | labels: {y.shape[1]}")

    model = build_mlp(input_dim=X_fit.shape[1], n_labels=y.shape[1], seed=args.seed)
    train_positives = y_fit.to_numpy().sum(axis=0)
    pos_weight = compute_pos_weight(y_fit, args.pos_weight_cap) if args.weighted else None
    if pos_weight is not None:
        print(f"pos_weight: min {pos_weight.min():.2f} | median {np.median(pos_weight):.2f} "
              f"| max {pos_weight.max():.2f} | at cap: {(pos_weight >= args.pos_weight_cap).sum()}")
    history = train_mlp(model, X_fit, y_fit, X_val, y_val,
                        loss_fn=build_loss(pos_weight), seed=args.seed)

    with timer() as t_pred:
        probs = predict_proba(model, X_val)
    y_pred = (probs >= args.threshold).astype(np.int8)

    shape_report = check_one_hot(y_pred, len(val_idx), y.shape[1])
    metrics = evaluate(y_val, y_pred)
    by_frequency = evaluate_by_frequency(y_val, y_pred, train_positives)
    true_per_issue = float(y_val.to_numpy().sum(axis=1).mean())

    print("\nOne-hot check passed:", json.dumps(shape_report))
    print(f"True labels per validation issue (mean): {true_per_issue:.3f}")
    print("Validation metrics @ threshold "
          f"{args.threshold}: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")
    print("Example row (indices of predicted labels):",
          np.flatnonzero(y_pred[0]).tolist())
    print("\nBy label frequency in the training rows:")
    print(pd.DataFrame(by_frequency).T.to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "threshold": args.threshold,
        "rows": {"fit": len(fit_idx), "validation": len(val_idx), "test_held_out": len(_test_idx)},
        "tfidf_features": X_fit.shape[1],
        "model": {"hidden_sizes": [m.out_features for m in model.net if hasattr(m, "out_features")][:-1],
                  "dropout": next(m.p for m in model.net if hasattr(m, "p"))},
        "one_hot_check": shape_report,
        "true_labels_per_issue": round(true_per_issue, 3),
        "validation_metrics": metrics,
        "validation_by_label_frequency": by_frequency,
        "weighted": args.weighted,
        "pos_weight_cap": args.pos_weight_cap if args.weighted else None,
        "predict_seconds": t_pred["seconds"],
        **history,
    }
    run_name = "weighted_run" if args.weighted else "first_run"
    path = RESULTS_DIR / f"{run_name}_seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()