import argparse
import json
from pathlib import Path

import pandas as pd

from db_config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"
TABLE_CSV = RESULTS_DIR / "results_table.csv"
TABLE_MD = RESULTS_DIR / "results_table.md"
DEFAULT_RUN = RESULTS_DIR / "mlp" / "first_run_seed42.json"

COLUMNS = ["model", "precision", "recall", "f1", "evaluated_on", "threshold", "source"]
METRICS = ["precision", "recall", "f1"]

BASELINE = {
    "model": "SkillScope RF + TF-IDF (paper)",
    "precision": 0.908,
    "recall": 0.876,
    "f1": 0.889,
    "evaluated_on": "paper's 80/20 split of the MLSMOTE-augmented data",
    "threshold": "",
    "source": "Carter et al., SkillScope, NLBSE 2025",
}

NOTE = (
    "The baseline numbers come from the paper and were measured on a split made "
    "after MLSMOTE oversampling, with TF-IDF fit on all rows; most test rows are "
    "synthetic and many have near-copies in training (see the baseline flaws "
    "notes). Our models use a leak-free split of the real issues, so the two are "
    "not measured on the same data. Validation scores are for development only; "
    "the test split is used once, for the final model."
)


def load_table() -> pd.DataFrame:
    if TABLE_CSV.is_file():
        table = pd.read_csv(TABLE_CSV, dtype={"threshold": str}, keep_default_na=False)
    else:
        table = pd.DataFrame(columns=COLUMNS)
    table = table[table["model"] != BASELINE["model"]]
    return pd.concat([pd.DataFrame([BASELINE]), table], ignore_index=True)[COLUMNS]


def write_markdown(table: pd.DataFrame, path=TABLE_MD) -> None:
    base_f1 = BASELINE["f1"]
    lines = [
        "# Results",
        "",
        "Micro-averaged scores over all 217 labels.",
        "",
        "| Model | Precision | Recall | Micro-F1 | ΔF1 vs baseline | Threshold | Evaluated on | Source |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        is_base = row.model == BASELINE["model"]
        delta = "—" if is_base else f"{float(row.f1) - base_f1:+.3f}"
        lines.append(
            f"| {row.model} | {float(row.precision):.3f} | {float(row.recall):.3f} "
            f"| {float(row.f1):.3f} | {delta} | {row.threshold or '—'} "
            f"| {row.evaluated_on} | {row.source} |"
        )
    lines += ["", f"_{NOTE}_", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def record_result(model: str, metrics: dict, evaluated_on: str,
                  threshold=None, source: str = "") -> pd.DataFrame:
    """Add or replace one model's row, then rewrite the CSV and the markdown table."""
    if model == BASELINE["model"]:
        raise ValueError("The baseline row is fixed; use another model name.")
    row = {"model": model, **{m: round(float(metrics[m]), 4) for m in METRICS},
           "evaluated_on": evaluated_on,
           "threshold": "" if threshold is None else str(threshold),
           "source": source}
    table = load_table()
    table = pd.concat([table[table["model"] != model], pd.DataFrame([row])],
                      ignore_index=True)[COLUMNS]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_CSV, index=False)
    write_markdown(table)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN, help="run JSON from run_mlp.py")
    ap.add_argument("--model", default="MLP (first run)", help="row name in the table")
    args = ap.parse_args()

    run_path = args.run.resolve()
    run = json.loads(run_path.read_text(encoding="utf-8"))
    rows = run["rows"]
    table = record_result(
        args.model,
        run["validation_metrics"],
        evaluated_on=(f"validation, {rows['validation']} issues "
                      f"(leak-free split, seed {run['seed']})"),
        threshold=run["threshold"],
        source=(run_path.relative_to(REPO_ROOT).as_posix()
                if run_path.is_relative_to(REPO_ROOT) else run_path.name),
    )
    print(table[["model", *METRICS]].to_string(index=False))
    print(f"Saved to: {TABLE_CSV} and {TABLE_MD}")


if __name__ == "__main__":
    main()