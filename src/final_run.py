import argparse
import json
from datetime import datetime

import numpy as np

from db_config import REPO_ROOT
from evaluate import evaluate, evaluate_by_frequency, timer
from results_table import BASELINE, record_result
from run_mlp import check_one_hot, load_fit_val
from search_mlp import SEARCH_DIR, train_config
from train_mlp import predict_proba

FINAL_DIR = REPO_ROOT / "results" / "final"
RESULTS_JSON = FINAL_DIR / "test_results.json"
PREDICTIONS = FINAL_DIR / "test_predictions.npz"
BASELINE_RUN_LOG = REPO_ROOT / "results" / "baseline_run_raw.json"
METRICS = ("precision", "recall", "f1")


def baseline_comparison(test_metrics: dict) -> dict:
    out = {
        "baseline": {k: BASELINE[k] for k in METRICS},
        "baseline_source": BASELINE["source"],
        "delta": {k: round(test_metrics[k] - BASELINE[k], 4) for k in METRICS},
    }
    if BASELINE_RUN_LOG.is_file():
        log = json.loads(BASELINE_RUN_LOG.read_text(encoding="utf-8"))
        out["baseline_run"] = {k: log.get(k) for k in
                               ("outcome", "runtime_seconds", "runtime_minutes", "peak_ram_gb")}
    else:
        out["baseline_run"] = None
    return out


def run(db, best: dict) -> tuple[dict, dict]:
    seed, threshold, config = best["seed"], best["threshold"], best["config"]
    data = load_fit_val(db, seed, include_test=True)
    print(f"Config: {json.dumps(config)} | threshold {threshold:g} | seed {seed}")

    with timer("retrain") as t_train:
        model, history = train_config(config, data, seed, verbose=True)

    # Validation check: must match the search exactly
    val_metrics = evaluate(data["y_val"], (predict_proba(model, data["X_val"]) >= threshold)
                           .astype(np.int8))
    expected = best["validation_metrics"]
    matches = all(np.isclose(val_metrics[k], expected[k], atol=1e-9) for k in METRICS)
    print(f"Validation F1 {val_metrics['f1']:.4f} vs search {expected['f1']:.4f} "
          f"-> {'matches' if matches else 'DOES NOT MATCH'}")

    # Test set
    with timer() as t_pred:
        probs = predict_proba(model, data["X_test"])
    y_pred = (probs >= threshold).astype(np.int8)
    shape_report = check_one_hot(y_pred, len(data["test_idx"]), data["y"].shape[1])
    test_metrics = evaluate(data["y_test"], y_pred)
    by_frequency = evaluate_by_frequency(data["y_test"], y_pred,
                                         data["y_fit"].to_numpy().sum(axis=0))

    result = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "config": config,
        "threshold": threshold,
        "rows": {"fit": len(data["fit_idx"]), "validation": len(data["val_idx"]),
                 "test": len(data["test_idx"])},
        "validation_check": {"metrics": val_metrics, "search_metrics": expected,
                             "matches_search": bool(matches)},
        "one_hot_check": shape_report,
        "true_labels_per_issue": round(float(data["y_test"].to_numpy().sum(axis=1).mean()), 3),
        "test_metrics": test_metrics,
        "test_by_label_frequency": by_frequency,
        "comparison": baseline_comparison(test_metrics),
        "runtime": {"train_seconds": history["train_seconds"],
                    "retrain_total_seconds": t_train["seconds"],
                    "test_predict_seconds": t_pred["seconds"]},
        **{k: history[k] for k in ("epochs_run", "best_epoch", "stopped_early",
                                   "train_loss", "val_loss", "train_settings", "device")},
    }
    predictions = {"y_pred": y_pred, "probs": probs, "test_idx": data["test_idx"],
                   "labels": np.array(data["y"].columns, dtype=str)}
    return result, predictions


def print_summary(result: dict) -> None:
    t, c = result["test_metrics"], result["comparison"]
    print(f"\nTest ({result['rows']['test']} issues, threshold {result['threshold']:g}): "
          f"P {t['precision']:.4f} | R {t['recall']:.4f} | F1 {t['f1']:.4f}")
    print(f"Baseline (paper):  P {c['baseline']['precision']:.4f} | "
          f"R {c['baseline']['recall']:.4f} | F1 {c['baseline']['f1']:.4f}")
    print(f"Diff (ours - paper): P {c['delta']['precision']:+.4f} | "
          f"R {c['delta']['recall']:+.4f} | F1 {c['delta']['f1']:+.4f}")
    r = result["runtime"]
    print(f"Runtime: train {r['train_seconds']:.1f} s | test predict {r['test_predict_seconds']:.3f} s")
    if c["baseline_run"]:
        b = c["baseline_run"]
        print(f"Baseline run log: {b['outcome']} after {b['runtime_minutes']} min "
              f"(peak RAM {b['peak_ram_gb']} GB)")
    print("One-hot check:", json.dumps(result["one_hot_check"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="optional path to the database")
    ap.add_argument("--reproduce", action="store_true",
                    help="re-run and compare with the saved test results (saves nothing new)")
    args = ap.parse_args()

    best = json.loads((SEARCH_DIR / "best_config.json").read_text(encoding="utf-8"))
    if RESULTS_JSON.is_file() and not args.reproduce:
        raise SystemExit(f"{RESULTS_JSON} already exists: the test set has been used. "
                         "Use --reproduce to re-run and compare instead.")
    if args.reproduce and not RESULTS_JSON.is_file():
        raise SystemExit("Nothing to reproduce yet: run without --reproduce first.")

    result, predictions = run(args.db, best)
    print_summary(result)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.reproduce:
        saved = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        saved_pred = np.load(PREDICTIONS)["y_pred"]
        check = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "predictions_identical": bool(np.array_equal(saved_pred, predictions["y_pred"])),
            "metrics_identical": all(np.isclose(saved["test_metrics"][k],
                                                result["test_metrics"][k], atol=1e-12)
                                     for k in METRICS),
            "saved_test_metrics": saved["test_metrics"],
            "rerun_test_metrics": result["test_metrics"],
            "rerun_train_seconds": result["runtime"]["train_seconds"],
        }
        (FINAL_DIR / "reproduce_check.json").write_text(json.dumps(check, indent=2),
                                                        encoding="utf-8")
        ok = check["predictions_identical"] and check["metrics_identical"]
        print(f"\nReproduction: {'identical' if ok else 'DIFFERENT'} "
              f"(predictions {check['predictions_identical']}, "
              f"metrics {check['metrics_identical']})")
        return

    RESULTS_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(PREDICTIONS, **predictions)
    record_result(
        f"MLP (final, threshold {result['threshold']:g})", result["test_metrics"],
        evaluated_on=f"test, {result['rows']['test']} issues (used once)",
        threshold=f"{result['threshold']:g}",
        source="results/final/test_results.json",
    )
    print(f"Saved to: {FINAL_DIR}")


if __name__ == "__main__":
    main()