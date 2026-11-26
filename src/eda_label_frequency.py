import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from db_config import REPO_ROOT, load_competition_table
from make_split import N_METADATA_COLUMNS

OUT_DIR = REPO_ROOT / "results" / "eda"
RARE_CUTOFF = 10

BAR = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def label_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The 217 label columns as 0/1 (counts above 0 become 1, as in the baseline)."""
    return (df.iloc[:, N_METADATA_COLUMNS:].fillna(0) > 0).astype(np.int8)


def label_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """One row per label: positives (issues with the label) and share of all issues."""
    positives = label_matrix(df).sum(axis=0).sort_values(ascending=False)
    return pd.DataFrame({
        "label": positives.index,
        "positives": positives.to_numpy(),
        "share_of_issues": (positives / len(df)).round(4).to_numpy(),
    })


def plot_label_frequencies(freq: pd.DataFrame, n_issues: int, path) -> None:
    shown = freq[freq["positives"] > 0]
    n_zero = len(freq) - len(shown)
    n_rare = int((freq["positives"] < RARE_CUTOFF).sum())

    fig, ax = plt.subplots(figsize=(24, 8), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = range(len(shown))
    ax.bar(x, shown["positives"], width=0.8, color=BAR, linewidth=0)

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.7)  # so a label with a single positive still shows a bar
    ax.set_xlim(-1, len(shown))
    ax.set_xticks(list(x))
    ax.set_xticklabels(shown["label"], rotation=90, fontsize=5, color=INK_SECONDARY)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", which="both", colors=MUTED, labelsize=9, length=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    ax.axhline(RARE_CUTOFF, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(shown) - 0.5, RARE_CUTOFF * 1.12, f"{RARE_CUTOFF} positives",
            ha="right", va="bottom", fontsize=8, color=INK_SECONDARY)

    top = shown.iloc[0]
    ax.annotate(f"{top['positives']:,} issues ({top['share_of_issues']:.0%})",
                (0, top["positives"]), xytext=(6, 4), textcoords="offset points",
                fontsize=8, color=INK_SECONDARY)

    ax.set_ylabel("Issues with the label (log scale)", fontsize=10, color=INK_SECONDARY)
    fig.suptitle("Skill label frequency", x=0.01, ha="left", fontsize=15,
                 color=INK, fontweight="bold")
    subtitle = (f"{len(freq)} labels across {n_issues:,} issues, most to least frequent. "
                f"{n_rare} labels have fewer than {RARE_CUTOFF} positives")
    subtitle += f"; {n_zero} with 0 positives are not shown." if n_zero else "."
    ax.set_title(subtitle, loc="left", fontsize=10, color=INK_SECONDARY, pad=10)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="optional path to the database")
    args = ap.parse_args()

    df = load_competition_table(args.db)
    freq = label_frequencies(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    freq.to_csv(OUT_DIR / "label_frequency.csv", index=False)
    plot_label_frequencies(freq, len(df), OUT_DIR / "label_frequency.png")

    print(f"Issues: {len(df)} | labels: {len(freq)}")
    print(f"Most frequent: {freq.iloc[0]['label']} ({freq.iloc[0]['positives']})")
    print(f"Labels with < {RARE_CUTOFF} positives: {(freq['positives'] < RARE_CUTOFF).sum()} "
          f"| with 0 positives: {(freq['positives'] == 0).sum()}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()