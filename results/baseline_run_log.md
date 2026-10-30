# Baseline run log (T18/T19)

- Date/time: 2026-10-28, started 19:45
- Script: competition_baseline/main.py (README calls it rf_baseline.py)
- Repo commit: 49aa792
- Python 3.11.18, Windows 10 (build 26200), 31.5 GB RAM
- Rows after preparation: 7,154 (the DB's pull_requests table has 7,245)
- Label columns: 217 in the table; 142 after MultiLabelBinarizer
- MLSMOTE synthetic rows requested: 757,925
- Augmented rows: 772,233 (7,154 + 757,925 + 7,154 duplicate original rows)
- Train/test rows: 617,786 / 154,447
- Outcome: stopped at the 60-minute cap during Random Forest training
- Peak RAM: 23.14 GB; no metrics produced
- Not a crash: the wrapper terminated it on purpose (exit code 15)
- Raw data: results/baseline_run_raw.json, results/baseline_stdout.txt