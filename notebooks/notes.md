# Baseline Run Notes:

From a test run of the baseline, taken from the Baseline Run Log, to confirm number of rows and columns. There is no provided muber for the split for the competition,
nor distribution of the split, so we will be using the reported splits used in the original paper, despite those splits being used on an augmented dataset. 

- Rows after preparation: 7,154 (the DB's pull_requests table has 7,245)

- Label columns: 217 in the table; 142 after MultiLabelBinarizer

- MLSMOTE synthetic rows requested: 757,925

- Augmented rows: 772,233 (7,154 + 757,925 + 7,154 duplicate original rows)

- Train/test rows: 617,786 / 154,447

- Train/Test Split: 80/20

- No null counts

