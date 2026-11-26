import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from db_config import REPO_ROOT
from eda_label_frequency import GRID, INK, INK_SECONDARY, MUTED, SURFACE

DEFAULT_RUN = REPO_ROOT / "results" / "mlp" / "first_run_seed42.json"

TRAIN_COLOR = "#257de0"
VAL_COLOR = "#f47230"


def plot_loss(run: dict, path) -> None:
    train, val = run["train_loss"], run["val_loss"]
    if not val:
        raise ValueError("This run has no validation loss to plot.")
    epochs = range(1, len(train) + 1)
    best = run["best_epoch"]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(epochs, train, color=TRAIN_COLOR, linewidth=2, label="Training")
    ax.plot(epochs, val, color=VAL_COLOR, linewidth=2, label="Validation")
    ax.scatter([best], [val[best - 1]], s=40, color=VAL_COLOR,
               edgecolor=SURFACE, linewidth=2, zorder=3)

    ax.axvline(best, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"Best epoch {best}\nval loss {val[best - 1]:.4f}",
                (best, val[best - 1]), xytext=(-8, 28), textcoords="offset points",
                ha="right", fontsize=8, color=INK_SECONDARY)

    last = len(train)
    for values, name in ((train, "Training"), (val, "Validation")):
        ax.annotate(name, (last, values[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_SECONDARY)

    later = train[1:] + val[1:] or train + val
    top = max(later) * 1.08
    ax.set_ylim(min(train + val) * 0.95, top)
    off_scale = (f"\nEpoch 1 (train {train[0]:.3f}, val {val[0]:.3f}) runs above the axis."
                 if max(train[0], val[0]) > top else "")

    ax.set_xlim(0.5, last + max(2, last * 0.12))
    ax.set_xlabel("Epoch", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Binary cross-entropy loss", fontsize=10, color=INK_SECONDARY)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.suptitle("MLP training and validation loss", x=0.01, ha="left",
                 fontsize=14, color=INK, fontweight="bold")
    stop = "early stop" if run.get("stopped_early") else "no early stop"
    ax.set_title(f"{last} epochs ({stop}); weights from epoch {best} are kept. "
                 f"Seed {run.get('seed')}, {run['rows']['fit']:,} training issues."
                 f"{off_scale}",
                 loc="left", fontsize=9, color=INK_SECONDARY, pad=8)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN, help="run JSON from run_mlp.py")
    args = ap.parse_args()

    run = json.loads(args.run.read_text(encoding="utf-8"))
    out = args.run.with_name(f"{args.run.stem}_loss.png")
    plot_loss(run, out)
    print(f"Best epoch {run['best_epoch']} of {len(run['train_loss'])} "
          f"| saved to: {out}")


if __name__ == "__main__":
    main()