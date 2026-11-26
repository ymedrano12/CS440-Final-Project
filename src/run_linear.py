import argparse
import json
import warnings
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from db_config import REPO_ROOT, load_competition_table
from evaluate import evaluate, timer
from make_split import load_split, split_validation
from preprocess import build_tfidf
from results_table import record_result
from run_mlp import check_one_hot, issue_texts, label_matrix

RESULTS_DIR = REPO_ROOT / "results" / "linear"

MODELS = {
    "logreg": "Logistic regression (one-vs-rest)",
    "svm": "Linear SVM (one-vs-rest)",
}


def build_model(name: str, C: float, seed: int) -> OneVsRestClassifier:
    if name == "logreg":
        base = LogisticRegression(C=C, solver="liblinear", max_iter=1000, random_state=seed)
    elif name == "svm":
        base = LinearSVC(C=C, max_iter=5000, random_state=seed)
    else:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
    return OneVsRestClassifier(base, n_jobs=-1)


def predict_labels(model: OneVsRestClassifier, name: str, X) -> tuple[np.ndarray, str]:
    if name == "logreg":
        return (model.predict_proba(X) >= 0.5).astype(np.int8), "0.5"
    return model.predict(X).astype(np.int8), "0 (SVM decision value)"


def run_one(name, C, seed, X_fit, y_fit, X_val, y_val) -> dict:
    model = build_model(name, C, seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Label .* is present in all training examples")
        with timer(f"{name} train") as t_train:
            model.fit(X_fit, y_fit)
    with timer() as t_pred:
        y_pred, threshold = predict_labels(model, name, X_val)

    shape_report = check_one_hot(y_pred, X_val.shape[0], y_fit.shape[1])
    metrics = evaluate(y_val, y_pred)
    print(f"{MODELS[name]}: {json.dumps({k: round(v, 4) for k, v in metrics.items()})} "
          f"| labels per issue {shape_report['mean_labels_per_issue']} "
          f"| never predicted {shape_report['labels_never_predicted']}")
    return {
        "model": MODELS[name],
        "C": C,
        "threshold": threshold,
        "one_hot_check": shape_report,
        "validation_metrics": metrics,
        "train_seconds": t_train["seconds"],
        "predict_seconds": t_pred["seconds"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="optional path to the database")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=[*MODELS, "both"], default="both")
    ap.add_argument("--C", type=float, default=1.0, help="inverse regularization strength")
    args = ap.parse_args()

    df = load_competition_table(args.db)
    texts, y = issue_texts(df), label_matrix(df)
    train_idx, _test_idx = load_split(df, args.seed)
    fit_idx, val_idx = split_validation(df, train_idx, seed=args.seed)
    _, X_fit, X_val = build_tfidf(texts, fit_idx, val_idx)
    y_fit = y.iloc[fit_idx].to_numpy(dtype=np.int64)
    y_val = y.iloc[val_idx].to_numpy(dtype=np.int64)
    print(f"Rows: fit {len(fit_idx)} | validation {len(val_idx)} | "
          f"TF-IDF features: {X_fit.shape[1]} | labels: {y.shape[1]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = list(MODELS) if args.model == "both" else [args.model]
    for name in names:
        result = run_one(name, args.C, args.seed, X_fit, y_fit, X_val, y_val)
        out = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "rows": {"fit": len(fit_idx), "validation": len(val_idx),
                     "test_held_out": len(_test_idx)},
            "tfidf_features": X_fit.shape[1],
            "true_labels_per_issue": round(float(y_val.sum(axis=1).mean()), 3),
            **result,
        }
        path = RESULTS_DIR / f"{name}_seed{args.seed}.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")

        record_result(
            result["model"], result["validation_metrics"],
            evaluated_on=(f"validation, {len(val_idx)} issues "
                          f"(leak-free split, seed {args.seed})"),
            threshold=result["threshold"],
            source=path.relative_to(REPO_ROOT).as_posix(),
        )
    print(f"Saved to: {RESULTS_DIR} and results/results_table.md")


if __name__ == "__main__":
    main()