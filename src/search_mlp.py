import argparse
import itertools
import json
import random

import pandas as pd

from mlp import build_loss, build_mlp, compute_pos_weight
from results_table import record_result
from run_mlp import RESULTS_DIR, load_fit_val
from threshold_sweep import choose, sweep
from train_mlp import predict_proba, train_mlp

SEARCH_DIR = RESULTS_DIR / "search"
RESULTS_CSV = SEARCH_DIR / "search_results.csv"

SEARCH_SPACE = {
    "hidden_sizes": [(256,), (512,), (1024,), (512, 256)],
    "dropout": [0.3, 0.5],
    "lr": [1e-3, 3e-4],
    "weight_decay": [0.0, 1e-5, 1e-4],
    "batch_size": [32, 64],
    "pos_weight_cap": [None, 3.0],  # None = unweighted loss
}
MODEL_KEYS = ("hidden_sizes", "dropout")
TRAIN_KEYS = ("lr", "weight_decay", "batch_size")


def all_configs() -> list[dict]:
    keys = list(SEARCH_SPACE)
    return [dict(zip(keys, values)) for values in itertools.product(*SEARCH_SPACE.values())]


def config_key(config: dict) -> str:
    return json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in config.items()}, sort_keys=True)


def run_config(config: dict, data: dict, seed: int) -> dict:
    model = build_mlp(input_dim=data["X_fit"].shape[1], n_labels=data["y"].shape[1],
                      seed=seed, **{k: config[k] for k in MODEL_KEYS})
    cap = config["pos_weight_cap"]
    pos_weight = None if cap is None else compute_pos_weight(data["y_fit"], cap)
    history = train_mlp(model, data["X_fit"], data["y_fit"], data["X_val"], data["y_val"],
                        loss_fn=build_loss(pos_weight), seed=seed, verbose=False,
                        **{k: config[k] for k in TRAIN_KEYS})

    probs = predict_proba(model, data["X_val"])
    curve = sweep("config", probs, data["y_val"].to_numpy())
    best = choose(curve)
    at_default = curve.loc[(curve["threshold"] - 0.5).abs().idxmin()]
    return {
        "config": config_key(config),
        **{k: (str(list(v)) if isinstance(v, tuple) else v) for k, v in config.items()},
        "epochs_run": history["epochs_run"],
        "best_epoch": history["best_epoch"],
        "train_seconds": history["train_seconds"],
        "threshold": float(best["threshold"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "f1_at_0.5": float(at_default["f1"]),
        "labels_per_issue": float(best["labels_per_issue"]),
        "labels_predicted": int(best["labels_predicted"]),
        "validation_rows": len(data["val_idx"]),
    }


def pick_configs(n_configs: int | None, seed: int) -> list[dict]:
    configs = all_configs()
    if n_configs is None or n_configs >= len(configs):
        return configs
    return random.Random(seed).sample(configs, n_configs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="optional path to the database")
    ap.add_argument("--seed", type=int, default=42)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--n-configs", type=int, default=60,
                       help="random sample of the grid (seeded, so the same sample each time)")
    group.add_argument("--all", action="store_true", help="run the full grid")
    args = ap.parse_args()

    configs = pick_configs(None if args.all else args.n_configs, args.seed)
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(RESULTS_CSV)["config"]) if RESULTS_CSV.is_file() else set()
    todo = [c for c in configs if config_key(c) not in done]
    print(f"Grid: {len(all_configs())} configs | this search: {len(configs)} "
          f"| already done: {len(configs) - len(todo)} | to run: {len(todo)}")

    if todo:
        data = load_fit_val(args.db, args.seed)
        for i, config in enumerate(todo, 1):
            row = run_config(config, data, args.seed)
            pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", index=False,
                                       header=not RESULTS_CSV.is_file())
            print(f"[{i}/{len(todo)}] F1 {row['f1']:.4f} (P {row['precision']:.3f}, "
                  f"R {row['recall']:.3f}) @ {row['threshold']:g} | "
                  f"{row['train_seconds']:.1f}s | {row['config']}")

    results = pd.read_csv(RESULTS_CSV, keep_default_na=False)
    results = results[results["config"].isin({config_key(c) for c in configs})]
    best = choose(results)
    top = results.sort_values("f1", ascending=False).head(5)
    print("\nTop 5 by validation micro-F1:")
    print(top[["f1", "precision", "recall", "threshold", *SEARCH_SPACE]].to_string(index=False))

    best_config = json.loads(best["config"])
    out = {
        "config": best_config,
        "threshold": float(best["threshold"]),
        "validation_metrics": {k: float(best[k]) for k in ("precision", "recall", "f1")},
        "f1_at_0.5": float(best["f1_at_0.5"]),
        "best_epoch": int(best["best_epoch"]),
        "seed": args.seed,
        "configs_compared": len(results),
        "rule": "best validation micro-F1 (each config at its own best threshold); "
                "within the F1 tolerance, highest recall",
    }
    (SEARCH_DIR / "best_config.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    record_result(
        f"MLP (best of search, threshold {best['threshold']:g})", out["validation_metrics"],
        evaluated_on=f"validation, {int(best['validation_rows'])} issues "
                     "(config and threshold picked here)",
        threshold=f"{best['threshold']:g}",
        source="results/mlp/search/best_config.json",
    )
    print(f"\nBest config: {json.dumps(best_config)}")
    print(f"Threshold {best['threshold']:g} | P {best['precision']:.4f} "
          f"R {best['recall']:.4f} F1 {best['f1']:.4f} ({len(results)} configs compared)")
    print(f"Saved to: {SEARCH_DIR}")


if __name__ == "__main__":
    main()