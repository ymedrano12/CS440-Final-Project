import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from db_config import load_competition_table
from eda_label_frequency import (BAR, GRID, INK, INK_SECONDARY, MUTED, OUT_DIR,
                                 SURFACE, label_matrix)


def labels_per_issue(df: pd.DataFrame) -> pd.Series:
    return label_matrix(df).sum(axis=1).rename("labels_per_issue")


def count_table(per_issue: pd.Series) -> pd.DataFrame:
    counts = per_issue.value_counts().reindex(range(per_issue.max() + 1), fill_value=0)
    return pd.DataFrame({
        "labels_per_issue": counts.index,
        "issues": counts.to_numpy(),
        "share_of_issues": (counts / len(per_issue)).round(4).to_numpy(),
    })


def plot_labels_per_issue(table: pd.DataFrame, per_issue: pd.Series, path) -> None:
    mean, median = per_issue.mean(), per_issue.median()
    n_zero = int((per_issue == 0).sum())

    fig, ax = plt.subplots(figsize=(14, 6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(table["labels_per_issue"], table["issues"], width=0.85, color=BAR, linewidth=0)

    max_count = int(table["labels_per_issue"].max())
    ax.set_xlim(-0.8, max_count + 0.8)
    ax.set_ylim(0, table["issues"].max() * 1.12)  # headroom for the mean label
    if max_count <= 30:
        ax.set_xticks(range(max_count + 1))  # one tick under every bar
    else:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="both", colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    ax.axvline(mean, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(mean, ax.get_ylim()[1], f" mean {mean:.1f}", ha="left", va="top",
            fontsize=8, color=INK_SECONDARY)

    ax.set_xlabel("Labels on the issue", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Issues", fontsize=10, color=INK_SECONDARY)
    fig.suptitle("Labels per issue", x=0.01, ha="left", fontsize=15,
                 color=INK, fontweight="bold")
    subtitle = (f"{len(per_issue):,} issues. Mean {mean:.1f}, median {median:.0f}, "
                f"max {per_issue.max()} labels; {n_zero:,} issues have no label.")
    ax.set_title(subtitle, loc="left", fontsize=10, color=INK_SECONDARY, pad=10)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="optional path to the database")
    args = ap.parse_args()

    df = load_competition_table(args.db)
    per_issue = labels_per_issue(df)
    table = count_table(per_issue)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "labels_per_issue.csv", index=False)
    plot_labels_per_issue(table, per_issue, OUT_DIR / "labels_per_issue.png")

    print(f"Issues: {len(per_issue)}")
    print(f"Labels per issue: mean {per_issue.mean():.2f} | median {per_issue.median():.0f} "
          f"| min {per_issue.min()} | max {per_issue.max()}")
    print(f"Issues with no label: {(per_issue == 0).sum()}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()