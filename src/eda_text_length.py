import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from db_config import load_competition_table
from eda_label_frequency import BAR, GRID, INK, INK_SECONDARY, MUTED, OUT_DIR, SURFACE
from make_split import TEXT_COLUMNS

FIELDS = {"title": TEXT_COLUMNS[0], "description": TEXT_COLUMNS[1]}


def word_count(text: pd.Series) -> pd.Series:
    return text.fillna("").astype(str).str.split().str.len()


def text_lengths(df: pd.DataFrame) -> pd.DataFrame:
    lengths = pd.DataFrame({name: word_count(df[col]) for name, col in FIELDS.items()})
    lengths["combined"] = lengths["title"] + lengths["description"]
    return lengths


def length_summary(lengths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in lengths.columns:
        words = lengths[field]
        rows.append({
            "field": field,
            "issues": len(words),
            "empty": int((words == 0).sum()),
            "mean": round(float(words.mean()), 1),
            "median": float(words.median()),
            "p90": float(words.quantile(0.90)),
            "p99": float(words.quantile(0.99)),
            "max": int(words.max()),
        })
    return pd.DataFrame(rows)


def plot_text_length(combined: pd.Series, path) -> None:
    shown = combined[combined > 0]
    n_empty = len(combined) - len(shown)
    median, p99 = combined.median(), combined.quantile(0.99)
    edges = np.unique(np.round(np.logspace(0, np.log10(shown.max() + 1), 40)))
    bins = np.append(edges, edges[-1] + 1) - 0.5

    fig, ax = plt.subplots(figsize=(14, 6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hist(shown, bins=bins, color=BAR, rwidth=0.88, linewidth=0)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="both", which="both", colors=MUTED, labelsize=9, length=0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    ax.axvline(median, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(median, ax.get_ylim()[1], f" median {median:,.0f} words", ha="left", va="top",
            fontsize=8, color=INK_SECONDARY)

    ax.set_xlabel("Words in title + description (log scale)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Issues", fontsize=10, color=INK_SECONDARY)
    fig.suptitle("Issue text length", x=0.01, ha="left", fontsize=15,
                 color=INK, fontweight="bold")
    subtitle = (f"{len(combined):,} issues. Median {median:,.0f} words, 99% of issues are "
                f"under {p99:,.0f}, longest {combined.max():,}")
    subtitle += f"; {n_empty:,} issues with no text are not shown." if n_empty else "."
    ax.set_title(subtitle, loc="left", fontsize=10, color=INK_SECONDARY, pad=10)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="optional path to the database")
    args = ap.parse_args()

    df = load_competition_table(args.db)
    lengths = text_lengths(df)
    summary = length_summary(lengths)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "text_length_summary.csv", index=False)
    plot_text_length(lengths["combined"], OUT_DIR / "text_length.png")

    print(f"Issues: {len(df)} (lengths in words)")
    print(summary.to_string(index=False))
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()