# Results

Micro-averaged scores over all 217 labels.

| Model | Precision | Recall | Micro-F1 | ΔF1 vs baseline | Threshold | Evaluated on | Source |
|---|---|---|---|---|---|---|---|
| SkillScope RF + TF-IDF (paper) | 0.908 | 0.876 | 0.889 | — | — | paper's 80/20 split of the MLSMOTE-augmented data | Carter et al., SkillScope, NLBSE 2025 |
| MLP (first run) | 0.697 | 0.387 | 0.497 | -0.392 | 0.5 | validation, 308 issues (leak-free split, seed 42) | results/mlp/first_run_seed42.json |
| Logistic regression (one-vs-rest) | 0.728 | 0.336 | 0.460 | -0.429 | 0.5 | validation, 308 issues (leak-free split, seed 42) | results/linear/logreg_seed42.json |
| Linear SVM (one-vs-rest) | 0.793 | 0.214 | 0.338 | -0.551 | 0 (SVM decision value) | validation, 308 issues (leak-free split, seed 42) | results/linear/svm_seed42.json |

_The baseline numbers come from the paper and were measured on a split made after MLSMOTE oversampling, with TF-IDF fit on all rows; most test rows are synthetic and many have near-copies in training (see the baseline flaws notes). Our models use a leak-free split of the real issues, so the two are not measured on the same data. Validation scores are for development only; the test split is used once, for the final model._
