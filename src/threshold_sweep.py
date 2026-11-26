import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_label_frequency import GRID, INK, INK_SECONDARY, MUTED, SURFACE
from evaluate import evaluate
from results_table import record_result
from run_mlp import RESULTS_DIR

DEFAULT_RUNS = ["first_run_seed42", "weighted_run_seed42"]
RUN_LABELS = {"first_run": "Unweighted MLP", "weighted_run": "Weighted MLP"}
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.025), 3)
DEFAULT_THRESHOLD = 0.5
F1_TOLERANCE = 0.005  # F1 differences this small are noise on ~300 issues

METRIC_COLORS = {"precision": "#257de0", "recall": "#f47230", "f1": "#00b894"}
METRIC_NAMES = {"precision": "Precision", "recall": "Recall", "f1": "Micro-F1"}


def run_label(run: str) -> str:
    return RUN_LABELS.get(run.rsplit("_seed", 1)[0], run)


def load_probs(run: str) -> tuple[np.ndarray, np.ndarray]:
    path = RESULTS_DIR / f"{run}_val_probs.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Re-run run_mlp.py for this run first.")
    data = np.load(path)
    return data["probs"], data["y_val"]


def sweep(run: str, probs: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        y_pred = (probs >= t).astype(np.int8)
        rows.append({
            "run": run,
            "threshold": float(t),
            **evaluate(y_val, y_pred),
            "labels_per_issue": round(float(y_pred.sum(axis=1).mean()), 3),
            "labels_predicted": int((y_pred.sum(axis=0) > 0).sum()),
        })
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame) -> pd.Series:
    best_f1 = table["f1"].max()
    candidates = table[table["f1"] >= best_f1 - F1_TOLERANCE]
    return candidates.sort_values(["recall", "threshold"], ascending=False).iloc[0]


def spread_labels(values: dict, min_gap: float = 0.06) -> dict:
    order = sorted(values, key=values.get)
    placed = {}
    for name in order:
        y = values[name]
        if placed:
            y = max(y, max(placed.values()) + min_gap)
        placed[name] = y
    return placed


def plot_sweep(table: pd.DataFrame, chosen: pd.Series, path) -> None:
    runs = list(dict.fromkeys(table["run"]))
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6), dpi=200,
                             sharey=True, facecolor=SURFACE, squeeze=False)
    for ax, run in zip(axes[0], runs):
        part = table[table["run"] == run]
        ax.set_facecolor(SURFACE)
        for metric, color in METRIC_COLORS.items():
            ax.plot(part["threshold"], part[metric], color=color, linewidth=2,
                    label=METRIC_NAMES[metric])
        x_end = part["threshold"].iloc[-1]
        ends = spread_labels({m: part[m].iloc[-1] for m in METRIC_COLORS})
        for metric, y in ends.items():
            ax.annotate(METRIC_NAMES[metric], (x_end, y), xytext=(5, 0),
                        textcoords="offset points", va="center",
                        fontsize=8, color=INK_SECONDARY)

        ax.axvline(DEFAULT_THRESHOLD, color=GRID, linewidth=1)
        ax.text(DEFAULT_THRESHOLD, 1.0, " default 0.5", fontsize=7, color=MUTED, va="top")
        if run == chosen["run"]:
            ax.axvline(chosen["threshold"], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            ax.scatter([chosen["threshold"]], [chosen["f1"]], s=40, color=METRIC_COLORS["f1"],
                       edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"Chosen {chosen['threshold']:g}\nF1 {chosen['f1']:.3f}, "
                        f"recall {chosen['recall']:.3f}",
                        (chosen["threshold"], chosen["f1"]), xytext=(8, -34),
                        textcoords="offset points", fontsize=8, color=INK_SECONDARY,
                        bbox={"boxstyle": "round,pad=0.25", "fc": SURFACE, "ec": "none"})

        ax.set_title(run_label(run), loc="left", fontsize=11, color=INK, pad=6)
        ax.set_xlim(THRESHOLDS[0], THRESHOLDS[-1] + 0.17)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Global threshold", fontsize=9, color=INK_SECONDARY)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0][0].set_ylabel("Validation score (micro)", fontsize=9, color=INK_SECONDARY)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.875), ncol=3,
               frameon=False, fontsize=8, labelcolor=INK_SECONDARY, handlelength=1.8)

    fig.suptitle("Threshold sweep on the validation set", x=0.01, y=0.99, ha="left",
                 fontsize=14, color=INK, fontweight="bold")
    fig.text(0.01, 0.905, "One cut-off applied to all 217 labels. Picked by best micro-F1, "
             f"then highest recall within {F1_TOLERANCE} F1.", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                    help="run names in results/mlp/ (without _val_probs.npz)")
    args = ap.parse_args()

    table = pd.concat([sweep(run, *load_probs(run)) for run in args.runs], ignore_index=True)
    chosen = choose(table)
    table.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
    plot_sweep(table, chosen, RESULTS_DIR / "threshold_sweep.png")

    for run in args.runs:
        part = table[table["run"] == run]
        at_default = part.loc[np.isclose(part["threshold"], DEFAULT_THRESHOLD)].iloc[0]
        best = part.loc[part["f1"].idxmax()]
        print(f"{run_label(run)}: F1 {at_default['f1']:.4f} at 0.5 -> best "
              f"{best['f1']:.4f} at {best['threshold']:g} "
              f"(P {best['precision']:.4f}, R {best['recall']:.4f})")

    metrics = {k: float(chosen[k]) for k in ("precision", "recall", "f1")}
    out = {
        "run": chosen["run"],
        "threshold": float(chosen["threshold"]),
        "validation_metrics": metrics,
        "labels_per_issue": float(chosen["labels_per_issue"]),
        "labels_predicted": int(chosen["labels_predicted"]),
        "rule": f"best validation micro-F1; within {F1_TOLERANCE} of it, highest recall",
        "thresholds_tried": [float(t) for t in THRESHOLDS],
    }
    (RESULTS_DIR / "chosen_threshold.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    record_result(
        f"{run_label(chosen['run'])} (threshold {chosen['threshold']:g})", metrics,
        evaluated_on=f"validation, {len(load_probs(chosen['run'])[1])} issues (threshold picked here)",
        threshold=f"{chosen['threshold']:g}",
        source="results/mlp/chosen_threshold.json",
    )
    print(f"\nChosen: {run_label(chosen['run'])} at {chosen['threshold']:g} | "
          f"P {metrics['precision']:.4f} R {metrics['recall']:.4f} F1 {metrics['f1']:.4f}")
    print(f"Saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()