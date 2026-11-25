import argparse
import json

import pandas as pd

from db_config import load_competition_table
from eda_label_frequency import OUT_DIR
from eda_text_length import FIELDS, text_lengths


def empty_text_table(df: pd.DataFrame, lengths: pd.DataFrame) -> pd.DataFrame:
    missing = {name: df[col].isna() for name, col in FIELDS.items()}
    missing["combined"] = missing["title"] & missing["description"]

    rows = []
    for field in ["title", "description", "combined"]:
        empty = lengths[field] == 0
        rows.append({
            "field": field,
            "issues": len(df),
            "missing": int(missing[field].sum()),
            "blank": int((empty & ~missing[field]).sum()),
            "empty": int(empty.sum()),
            "empty_rate": round(float(empty.mean()), 4),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="optional path to the database")
    args = ap.parse_args()

    df = load_competition_table(args.db)
    lengths = text_lengths(df)
    table = empty_text_table(df, lengths)
    title_only = int(((lengths["title"] > 0) & (lengths["description"] == 0)).sum())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "empty_text.csv", index=False)
    summary = {
        "issues": len(df),
        "fields": table.set_index("field").drop(columns="issues").to_dict(orient="index"),
        "issues_with_title_only": title_only,
    }
    (OUT_DIR / "empty_text_summary.json").write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")

    print(f"Issues: {len(df)}")
    print(table.assign(empty_rate=table["empty_rate"].map("{:.2%}".format)).to_string(index=False))
    print(f"Issues with a title but no description: {title_only} ({title_only / len(df):.2%})")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()