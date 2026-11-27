# CS440-Final-Project
 
 
## NLBSE 2027 Tool Competition: Skill Classification Competition
 
Official Page: https://nlbse2027.github.io/tools/   
GitHub Repository: https://github.com/nlbse2027/skill-classification  
Overleaf Project Report: https://www.overleaf.com/project/6ab13d836d0ae3067fcfad38  
Word Document Project Roadmap: [https://colostate-my.sharepoint.com](https://colostate-my.sharepoint.com/:w:/r/personal/ymedrano_colostate_edu/_layouts/15/Doc.aspx?sourcedoc=%7BE5DBD08F-A02F-42EB-817B-85636564B0AA%7D&file=CS440%20Final%20Project%20Roadmap.docx&fromShare=true&action=default&mobileredirect=true)  
**Our Project Repository: https://github.com/ymedrano12/CS440-Final-Project**
 
## About This Project
 
Team: Catrina Coons, Yulisa Medrano (CS440, Colorado State University)
 
We built a multi-label classifier that predicts which of the 217 SkillScope skill labels (31 domains, 186 subdomains) an issue needs, using only the issue's title and description. Our main model is a multi-layer perceptron (MLP) trained on TF-IDF features with binary cross-entropy loss and one sigmoid output per label. We compare it with one-vs-rest logistic regression and a one-vs-rest linear SVM, all built on the same split, features, and evaluation code.
 
Unlike the competition baseline, our pipeline is leak-free:
 
- The real issues are split **before** any feature fitting (70% fit / 10% validation / 20% test, seed 42). Issues with identical text are grouped so duplicates never land on both sides.
- TF-IDF is fit on the training rows only.
- No synthetic oversampling. Class imbalance is handled with per-label loss weights computed from the training rows.
- The threshold and hyperparameters are chosen on validation only. The test split was evaluated **once**.

Because the numbers are not directly comparable, and micro-F1, precision, and recall could not meet or exceed the baseline's numbers, this repository and the accompanying report will not be submitted to the competition. The main purpose of this project is to serve as a more exploratory look at this database and as the submission for the Fall 2026 CS440 Final Project. 

## Results
 
Micro-averaged precision, recall and F1 over all 217 labels.
 
| Model | Precision | Recall | Micro-F1 | Evaluated on |
|---|---|---|---|---|
| SkillScope RF + TF-IDF (paper) | 0.908 | 0.876 | 0.889 | paper's 80/20 split of MLSMOTE-augmented data |
| Logistic regression (one-vs-rest) | 0.728 | 0.336 | 0.460 | validation, 308 issues |
| Linear SVM (one-vs-rest) | 0.793 | 0.214 | 0.338 | validation, 308 issues |
| MLP, first run (threshold 0.5) | 0.697 | 0.387 | 0.497 | validation, 308 issues |
| MLP, best of search (threshold 0.425) | 0.488 | 0.629 | 0.549 | validation, 308 issues |
| **MLP, final (threshold 0.425)** | **0.502** | **0.635** | **0.561** | **test, 616 issues (used once)** |
 
Final model: one hidden layer of 1024 units, dropout 0.3, Adam (lr 0.001, weight decay 1e-4), batch size 64, label weights capped at 3, early stopping (best epoch 38 of 43). It trains in about 13 seconds on a CPU, and a second run gave identical predictions and scores (`results/final/reproduce_check.json`).
 
**These numbers are not directly comparable to the baseline.** The paper's baseline fits TF-IDF and MLSMOTE oversampling on the whole dataset before splitting, duplicates every original row, and ends up with a test set that is roughly 98% synthetic rows with near-copies in training. Our scores come from real, unseen issues. See `results/baseline_reference.md` and the baseline flaws notes for details. The full table, with every intermediate run, is in `results/results_table.md`.
 
On the test set, the model does well on common labels (F1 0.58 over the 67 labels with 100+ training positives) but almost never predicts medium-frequency labels and never predicts rare ones. 75 of the 217 labels have no training positives at all.
 
We also tried to re-run the baseline (`src/run_baseline_timed.py`). It requested about 758,000 synthetic rows, reached 23 GB of RAM, and was still training Random Forest when it was stopped at the 60-minute cap, so it produced no metrics (`results/baseline_run_log.md`).

 
## Repository Layout
 
```
competition_baseline/   Organizers' Random Forest baseline (unchanged)
data/
  splits/               split_seed42.csv (committed split) + split_seed42_meta.json
  (skillscope_data_v2.db goes here; not committed)
notebooks/              informal notes on the original baseline run
report/                 Report source
results/
  db_exploration/       Tables/views, schema, shape, DB summary
  eda/                  Label frequency, labels per issue, rare labels, issues per repo, text length, empty text
  linear/               Logistic regression and SVM results
  mlp/                  MLP runs, loss curves, threshold sweep
    search/             Hyperparameter search results and best config
  final/                Test results, test predictions, reproduction check
  baseline_*            Baseline reference numbers and run log
  results_table.csv/.md All models side by side
src/
  db_config.py          Database path (SKILLSCOPE_DB / --db / default) and loading
  explore_db.py         Tables, schema and row count
  eda_*.py              Exploratory data analysis charts and summaries
  preprocess.py         Text cleaning (copied from the baseline) and TF-IDF fit on training rows only
  make_split.py         Leak-free train/test split and the validation split
  evaluate.py           Micro P/R/F1, scores by label frequency, runtime timer
  mlp.py                MLP architecture, loss, per-label weights
  train_mlp.py          Training loop with early stopping
  run_mlp.py            Train one MLP (unweighted or --weighted)
  run_linear.py         Logistic regression and linear SVM
  plot_loss.py          Train/validation loss curve
  threshold_sweep.py    Global threshold sweep on validation
  search_mlp.py         Hyperparameter search on validation
  final_run.py          Retrain the best config and evaluate on test once
  results_table.py      Builds results/results_table.csv and .md
  run_baseline_timed.py Runs the competition baseline with a time and RAM limit
  train.py              Small Random Forest sanity check on 100 rows
```
 
## Setup
 
Python 3.11 is recommended (it matches the baseline's environment).
 
```console
git clone https://github.com/ymedrano12/CS440-Final-Project.git
cd CS440-Final-Project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
 
On Windows, activate the environment with `.venv\Scripts\activate` instead.
 
All commands below are run from the repository root.
 
## Database
 
1. Download `skillscope_data.db` from the [NLBSE Skill Competition dataset on Hugging Face](https://huggingface.co/datasets/NLBSE/SkillCompetition).
2. Tell the scripts where it is, in one of three ways (checked in this order):
   - pass `--db PATH` to any script that accepts it;
   - set the `SKILLSCOPE_DB` environment variable;
   - or save the file as `data/skillscope_data_v2.db` (the default location).
```console
# macOS / Linux
export SKILLSCOPE_DB=/path/to/skillscope_data.db
# Windows (PowerShell)
$env:SKILLSCOPE_DB = "C:\path\to\skillscope_data.db"
```
 
Check that the database is found:
 
```console
python src/db_config.py
```
 
This prints the path being used and the tables and views in the database. If the file is missing, the error says which source the path came from and how to fix it.
 
The version of the database we used has 3,070 issues and 217 label columns in `nlbse_tool_competition_data_by_issue`. The saved split checks the row count and row order against the database and stops with an error if they don't match.

Note: late into the semester of this project work, the database we were using was updated and re-provided to students using this competition for their project, which is why here it is denoted as 'v2'. Some notes about the database may be outdated from the original runs as a result. 
 
## Running the Code
 
The steps below reproduce every result in order. Each script saves its output under `results/` and prints where it saved it. All seeds default to 42.
 
**1. Explore the database and data**
 
```console
python src/explore_db.py
python src/eda_label_frequency.py
python src/eda_labels_per_issue.py
python src/eda_rare_labels.py
python src/eda_issues_per_repo.py
python src/eda_text_length.py
python src/eda_empty_text.py
```
 
**2. Split** (already committed in `data/splits/`; re-running gives the same split)
 
```console
python src/make_split.py
```
 
**3. First models** (validation only; the test split is not touched)
 
```console
python src/run_mlp.py                 # unweighted MLP
python src/run_mlp.py --weighted      # MLP with per-label loss weights
python src/plot_loss.py
python src/plot_loss.py --run results/mlp/weighted_run_seed42.json
python src/run_linear.py              # logistic regression and linear SVM
python src/threshold_sweep.py
```
 
**4. Hyperparameter search** (60 sampled configs)
 
```console
python src/search_mlp.py
```
 
Configs already in `results/mlp/search/search_results.csv` are skipped, so an interrupted search can be resumed by running the same command. To redo the search from scratch, move or rename that CSV first. Use `--all` to run the full 384-config grid.
 
**5. Final model on the test set**
 
```console
python src/final_run.py --reproduce
```
 
`final_run.py` evaluates the test set only once. Because `results/final/test_results.json` is already committed, running it without `--reproduce` stops with a message. `--reproduce` retrains the best config with the same seed, re-scores the test set, and checks that the predictions and scores match the saved ones; it writes only `results/final/reproduce_check.json`.
 
**Optional: competition baseline**
 
```console
python src/run_baseline_timed.py      # --minutes 60 --max-ram-percent 80 by default
```
 
Warning: this needs a lot of memory and time (see Results). It stops itself at the time or RAM limit and records what happened in `results/baseline_run_raw.json`.
 
## Expected Output
 
`python src/final_run.py --reproduce` should end with (training times will vary slightly by machine):
 
```
Rows: fit 2146 | validation 308 | test 616 (evaluated)
TF-IDF features: 1000 | labels: 217
Config: {"batch_size": 64, "dropout": 0.3, "hidden_sizes": [1024], "lr": 0.001, "pos_weight_cap": 3.0, "weight_decay": 0.0001} | threshold 0.425 | seed 42
epoch   1 | train loss 0.5465 | val loss 0.3084
...
Early stop: no val improvement for 5 epochs.
Best epoch: 38 | training time 12.5 s on cpu
Validation F1 0.5493 vs search 0.5493 -> matches
 
Test (616 issues, threshold 0.425): P 0.5021 | R 0.6347 | F1 0.5606
Baseline (paper):  P 0.9080 | R 0.8760 | F1 0.8890
Diff (ours - paper): P -0.4059 | R -0.2413 | F1 -0.3284
Runtime: train 12.5 s | test predict 0.009 s
Baseline run log: stopped: time limit reached after 60.0 min (peak RAM 23.14 GB)
One-hot check: {"shape": [616, 217], "values": [0, 1], "mean_labels_per_issue": 18.925, "issues_with_no_label": 0, "labels_never_predicted": 135}
 
Reproduction: identical (predictions True, metrics True)
```
 
The final prediction is a 616 × 217 matrix of 0s and 1s (one row per test issue, one column per skill label), saved in `results/final/test_predictions.npz` with the probabilities and label names.
 
Our results were produced on a CPU. With a CUDA GPU, PyTorch picks the GPU automatically and the numbers may differ slightly. To match ours exactly, hide the GPU (`CUDA_VISIBLE_DEVICES=""`, or `$env:CUDA_VISIBLE_DEVICES=""` in PowerShell).
 
---
 
# Tool Competition - Skill Classification
 
## Competition Overview
 
The 2027 competition consists of building and assessing multi‑label classifiers that predict, for each issue, the set of domains and sub‑domains representing the skills required to solve it.
 
 
## Dataset
 
We release a dataset mined from 7,245 merged pull requests across 11 popular Java repositories (57,206 source files; 59,644 methods; 13,097 classes) annotated with 217 skill labels composed by domain/sub‑domains You may find the dataset in a SQLite database, named `skillscope_data.db`, [here](https://huggingface.co/datasets/NLBSE/SkillCompetition). This dataset is an updated version of the one submitted in the original [SkillScope](#citing-relevant-work) paper.
 
### Ready-made Table
Inside the database you will find a table named `nlbse_tool_competition_data_by_issue` that joins each pull request’s textual and code‑context features with its canonical domain/sub‑domain labels per issue. There is also a view `vw_nlbse_tool_competition_data_by_file` that labels each filename/function associated with each issue. 
 
- The `nlbse_tool_competition_data_by_issue` table contains a column for each domain and subdomain with an integer count of the number of matching APIs found for that issue. A value greater than zero indicates that domain/subdomain is present in the issue.
- The `vw_nlbse_tool_competition_data_by_file` is present for convenience purposes only and will not be used in evaluation of the model.
### Additional Metadata/Input Data is Allowed
Your model is allowed to use input data that is outside what is given in the database. You may use additional GitHub APIs to fetch more metadata, or download relevant files in an issue for further analysis. However, you must not use any third-party classification engine or the outputs present in `skillscope_data.db` as direct inputs to the model. 
 
### Baselines
Your models will be compared against the SkillScope Random‑Forest + TF‑IDF baselines reported in the paper by evaluating the overall prediction metrics against the issue classifications as recorded in the `nlbse_tool_competition_data_by_issue` table. The models you create should return a multi-label classification encoded in an one-hot encoded vector. However, it is the metrics of your models which will be the subject to the evaluation instead of the specific model output. Random Forest Baseline is provided in the folder `competition_baseline` with README and requirements.txt.
 
### Goal
Train, tune and evaluate your models on the provided splits and improve at least one of precision, recall, or micro‑F1 while not decreasing the remaining metrics relative to the best baseline.
 
## Organizers
 
The skill classification competition is organized by: Fabio Santos (fabio.deabreusantos@colostate.edu), and Jacob Penney (jmp458@nau.edu)
 
## Submission Acceptance & Competition
 
Submissions will first be filtered to ensure they do not lower any of the three core metrics (precision, recall, micro‑F1) compared with the baseline. Among qualifying submissions, ranking is determined by the largest positive improvement in micro‑F1. Ties will be broken by (1) precision, then (2) runtime.
 
Submissions will be judged on:
- Clarity and completeness of the paper.
- Availability and reproducibility of code/tool (open‑source required).
- Correct reporting of metrics.
- Quality of documentation.
Accepted papers will appear in the **NLBSE ’27 proceedings**.
 
### Ranking Details
 
Participants submit a single multi‑label classifier. Rankings proceed in two stages:
 
1. Eligibility check: The submission must *not* reduce any of precision, recall, or micro‑F1 compared with the baseline.
2. Ordering: Qualifying submissions are ordered by the greatest positive ∆micro‑F1.
## Citing Relevant Work
 
### General Competition Citation:
 
```tex
@inproceedings{nlbse2027,
  author = {Mock, Moritz and Borsani, Thomas and Russo, Barbara and Santos, Fabio and and Penney, Jacob and Valenzuela{-}Toledo, Pablo and Kehrer, Timo and Panichella, Sebastiano },
  title={The NLBSE'27 Tool Competition and Challenge},
  booktitle={Proceedings of The 6th International Workshop on Natural Language-based Software Engineering (NLBSE'27)},
  year={2027}
}
```
 
### Skill Competition Citation:
```tex
@inproceedings{carter2025skillscope,
    title        = {SkillScope: A Tool to Predict Fine-Grained Skills Needed to Solve Issues on GitHub},
    author       = {Carter, Benjamin C. and Contreras, Jonathan Rivas and Llanes Villegas, Carlos A. and Acharya, Pawan and Utzerath, Jack and Farner, Adonijah O. and Jenkins, Hunter and Johnson, Dylan and Penney, Jacob and Steinmacher, Igor and Gerosa, Marco A. and Santos, Fabio},
    year         = 2025,
    booktitle    = {2025 IEEE/ACM International Workshop on Natural Language-Based Software Engineering (NLBSE)},
    volume       = {},
    number       = {},
    pages        = {9--12},
    doi          = {10.1109/NLBSE66842.2025.00007},
    keywords     = {Random Forest;Training;Java;Large language models;Semantics;Retrieval augmented generation;Machine learning;Open source software;Software engineering;Software development management;software engineering;skill categorization;open source software (OSS);machine learning;large language models}
}
```
```tex
@article{santos2023tag,
    title        = {Tag that issue: applying API-domain labels in issue tracking systems},
    author       = {Santos, Fabio and Vargovich, Joseph and Trinkenreich, Bianca and Santos, Italo and Penney, Jacob and Britto, Ricardo and Pimentel, Jo{\~a}o Felipe and Wiese, Igor and Steinmacher, Igor and Sarma, Anita and others},
    year         = 2023,
    journal      = {Empirical Software Engineering},
    publisher    = {Springer},
    volume       = 28,
    number       = 5,
    pages        = 116
}
```
```tex
@inproceedings{vargovich2023givemelabeledissues,
    title        = {Givemelabeledissues: An open source issue recommendation system},
    author       = {Vargovich, Joseph and Santos, Fabio and Penney, Jacob and Gerosa, Marco A and Steinmacher, Igor},
    year         = 2023,
    booktitle    = {2023 IEEE/ACM 20th International Conference on Mining Software Repositories (MSR)},
    pages        = {402--406},
    organization = {IEEE}
}
```
 
