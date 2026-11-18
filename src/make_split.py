import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from db_config import REPO_ROOT, get_db_path, load_competition_table

SPLIT_DIR = REPO_ROOT / "data" / "splits"
N_METADATA_COLUMNS = 9
TEXT_COLUMNS = ["issue text", "issue description"]
ID_COLUMNS = ["Repo Name", "PR #"]


def split_path(seed: int):
    return SPLIT_DIR / f"split_seed{seed}.csv"


def duplicate_groups(df: pd.DataFrame) -> np.ndarray:
    text = (
        df[TEXT_COLUMNS[0]].fillna("") + " " + df[TEXT_COLUMNS[1]].fillna("")
    ).str.lower().str.split().str.join(" ")
    return pd.factorize(text)[0]


def make_split(df: pd.DataFrame, test_size: float, seed: int) -> pd.Series:
    groups = duplicate_groups(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    split = pd.Series("train", index=range(len(df)), name="split")
    split.iloc[test_idx] = "test"
    return split


def label_report(df: pd.DataFrame, split: pd.Series) -> dict:
    labels = df.iloc[:, N_METADATA_COLUMNS:]
    positive = labels.fillna(0).to_numpy() > 0
    is_test = (split == "test").to_numpy()
    train_pos = positive[~is_test].sum(axis=0)
    test_pos = positive[is_test].sum(axis=0)
    names = labels.columns
    return {
        "label_count": len(names),
        "labels_with_no_positives_at_all": [
            n for n, a, b in zip(names, train_pos, test_pos) if a + b == 0
        ],
        "labels_missing_from_train": [
            n for n, a, b in zip(names, train_pos, test_pos) if a == 0 and b > 0
        ],
        "labels_missing_from_test": [
            n for n, a, b in zip(names, train_pos, test_pos) if b == 0 and a > 0
        ],
    }


def load_split(df: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    saved = pd.read_csv(split_path(seed))
    if len(saved) != len(df):
        raise ValueError(
            f"Split has {len(saved)} rows but the table has {len(df)}. "
            "Was the database changed? Re-run make_split.py."
        )
    current = df[ID_COLUMNS].reset_index(drop=True).astype(str)
    stored = saved[ID_COLUMNS].astype(str)
    if not current.equals(stored):
        raise ValueError("Row order of the table does not match the saved split.")

    is_test = (saved["split"] == "test").to_numpy()
    rows = saved["row_index"].to_numpy()
    return rows[~is_test], rows[is_test]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="optional path to the database")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = load_competition_table(args.db)
    split = make_split(df, args.test_size, args.seed)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    out = df[ID_COLUMNS].copy()
    out.insert(0, "row_index", range(len(df)))
    out["split"] = split.to_numpy()
    out.to_csv(split_path(args.seed), index=False)

    counts = split.value_counts().to_dict()
    groups = duplicate_groups(df)
    meta = {
        "database": get_db_path(args.db).name,
        "method": "GroupShuffleSplit on real issues, grouped by duplicate text",
        "seed": args.seed,
        "test_size": args.test_size,
        "total_rows": len(df),
        "train_rows": counts.get("train", 0),
        "test_rows": counts.get("test", 0),
        "duplicate_rows_grouped": int(len(df) - len(np.unique(groups))),
        **label_report(df, split),
    }
    meta_path = SPLIT_DIR / f"split_seed{args.seed}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Rows: {meta['total_rows']} | train: {meta['train_rows']} "
          f"| test: {meta['test_rows']} ({meta['test_rows'] / len(df):.1%})")
    print(f"Duplicate-text rows kept on one side: {meta['duplicate_rows_grouped']}")
    print(f"Labels: {meta['label_count']} | missing from train: "
          f"{len(meta['labels_missing_from_train'])} | missing from test: "
          f"{len(meta['labels_missing_from_test'])} | no positives at all: "
          f"{len(meta['labels_with_no_positives_at_all'])}")
    print(f"Saved to: {SPLIT_DIR}")


if __name__ == "__main__":
    main()