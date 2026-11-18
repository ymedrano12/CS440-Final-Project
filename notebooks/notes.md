# Baseline Run Notes:

From a test run of the baseline, taken from the Baseline Run Log, to confirm number of rows and columns. There is no provided muber for the split for the competition,
nor distribution of the split, so we will be using the reported splits used in the original paper, despite those splits being used on an augmented dataset. 

- Tables/views: name,type
files_changed,table
sqlite_sequence,table
api_file_register,table
repositories,table
pull_request_commits,table
pull_request_comments,table
settings,table
pull_requests,table
api_cache,table
function_cache,table
files_downloaded,table
sqlite_stat1,table
sqlite_stat4,table
vw_nlbse_tool_competition_data_by_file,view
vw_nlbse_tool_competition_data_by_issue,view
vw_classification_raw,view
classification_raw,table
nlbse_tool_competition_data_by_issue,table
 (DB re-issued by organizers, 11-17-26)

- Rows: 3070 (updated from new db)

- Labels: 217 (columns after the first 9 metadata columns)

- MLSMOTE synthetic rows requested: 757,925

- Augmented rows: 772,233 (7,154 + 757,925 + 7,154 duplicate original rows)

- Train/test rows: 617,786 / 154,447

- Train/Test Split: 80/20 (based on original paper, not provided)

- No null counts

