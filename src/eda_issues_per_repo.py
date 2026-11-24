import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from db_config import load_competition_table
from eda_label_frequency import BAR, GRID, INK, INK_SECONDARY, MUTED, OUT_DIR, SURFACE
from make_split import ID_COLUMNS

REPO_COLUMN = ID_COLUMNS[0]


def issues_per_repo(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[REPO_COLUMN].fillna("(missing)").value_counts()
    return pd.DataFrame({
        "repo": counts.index,
        "issues": counts.to_numpy(),
        "share_of_issues": (counts / len(df)).round(4).to_numpy(),
    })


def plot_issues_per_repo(table: pd.DataFrame, path) -> None:
    n_issues = int(table["issues"].sum())
    top = table.iloc[0]
    rows = table.iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, 0.45 * len(table) + 1.8), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(rows))
    ax.barh(y, rows["issues"], height=0.7, color=BAR, linewidth=0)

    ax.set_yticks(list(y))
    ax.set_yticklabels(rows["repo"], fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(0, table["issues"].max() * 1.18)  # room for the value labels
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="both", colors=MUTED, labelsize=9, length=0)
    ax.tick_params(axis="y", colors=INK_SECONDARY)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)

    for yi, (issues, share) in zip(y, zip(rows["issues"], rows["share_of_issues"])):
        ax.text(issues, yi, f"  {issues:,} ({share:.0%})", va="center", ha="left",
                fontsize=8, color=INK_SECONDARY)

    ax.set_xlabel("Issues", fontsize=10, color=INK_SECONDARY)
    fig.suptitle("Issues per repository", x=0.01, ha="left", fontsize=15,
                 color=INK, fontweight="bold")
    subtitle = (f"{n_issues:,} issues across {len(table)} repositories. "
                f"The largest, {top['repo']}, holds {top['share_of_issues']:.0%} of them.")
    ax.set_title(subtitle, loc="left", fontsize=10, color=INK_SECONDARY, pad=10)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="optional path to the database")
    args = ap.parse_args()

    df = load_competition_table(args.db)
    table = issues_per_repo(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "issues_per_repo.csv", index=False)
    plot_issues_per_repo(table, OUT_DIR / "issues_per_repo.png")

    print(f"Issues: {len(df)} | repositories: {len(table)}")
    print(table.to_string(index=False))
    print(f"Largest / smallest repository: {table['issues'].iloc[0] / table['issues'].iloc[-1]:.1f}x")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()