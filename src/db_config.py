from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ENV_VAR = "SKILLSCOPE_DB"
COMPETITION_TABLE = "nlbse_tool_competition_data_by_issue"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "skillscope_data.db"

def get_db_path(cli_path: str | None = None) -> Path:
    if cli_path:
        path, source = Path(cli_path), "command-line argument"
    elif(os.environ.get(ENV_VAR)):
        path, source = Path(os.environ[ENV_VAR]), f"{ENV_VAR} environment variable"
    else:
        path, source = DEFAULT_DB, "default location"

    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Database not found at: {path}\n"
            f"(path came from the {source})\n"
            f"Fix it by setting {ENV_VAR}, or ensure file is at {DEFAULT_DB}")
    
    return path

def connect(cli_path: str | None = None) -> sqlite3.Connection:
    path = get_db_path(cli_path)
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)

def load_competition_table(cli_path: str | None = None) -> pd.DataFrame:
    con = connect(cli_path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {COMPETITION_TABLE}", con)
    finally:
        con.close()

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Using database: {get_db_path(arg)}")
    con = connect(arg)
    print(pd.read_sql_query(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')",
            con,
        )
    )
    con.close()