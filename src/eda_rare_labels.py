import argparse
import json

from db_config import load_competition_table
from eda_label_frequency import OUT_DIR, RARE_CUTOFF, label_frequencies, label_matrix


def rare_label_summary(freq, n_issues: int, total_positives: int, cutoff: int) -> dict:
    rare = freq[freq["positives"] < cutoff]
    return {
        "cutoff": cutoff,
        "issues": n_issues,
        "labels": len(freq),
        "rare_labels": len(rare),
        "rare_share_of_labels": round(len(rare) / len(freq), 4),
        "labels_with_0_positives": int((freq["positives"] == 0).sum()),
        f"labels_with_1_to_{cutoff - 1}_positives": int(rare["positives"].between(1, cutoff - 1).sum()),
        "rare_share_of_all_positives": round(int(rare["positives"].sum()) / total_positives, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="optional path to the database")
    ap.add_argument("--cutoff", type=int, default=RARE_CUTOFF,
                    help=f"a label is rare below this many positives (default {RARE_CUTOFF})")
    args = ap.parse_args()

    df = load_competition_table(args.db)
    freq = label_frequencies(df)
    total_positives = int(label_matrix(df).to_numpy().sum())
    summary = rare_label_summary(freq, len(df), total_positives, args.cutoff)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    freq[freq["positives"] < args.cutoff].to_csv(OUT_DIR / "rare_labels.csv", index=False)
    out = {**summary}
    (OUT_DIR / "rare_labels_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Labels with fewer than {args.cutoff} positives: {summary['rare_labels']} "
          f"of {summary['labels']} ({summary['rare_share_of_labels']:.1%})")
    print(f"  with 0 positives: {summary['labels_with_0_positives']}")
    print(f"  with 1-{args.cutoff - 1} positives: "
          f"{summary[f'labels_with_1_to_{args.cutoff - 1}_positives']}")
    print(f"Rare labels hold {summary['rare_share_of_all_positives']:.2%} of all label positives")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()