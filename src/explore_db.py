import pandas as pd

from db_config import COMPETITION_TABLE, REPO_ROOT, connect

OUT_DIR = REPO_ROOT / "results" / "db_exploration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

con = connect()

# List tables and views
tables = pd.read_sql_query("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')",
                           con)
print("Tables and Views")
print(tables.to_string(index=False))
tables.to_csv(OUT_DIR / "tables_and_views.csv", index=False)


# Print the schema
schema = pd.read_sql_query(f"PRAGMA table_info({COMPETITION_TABLE})", con)
print(f"\n Schema of {COMPETITION_TABLE}")
print(f"Number of Columns: {len(schema)}")
print(schema[["cid", "name", "type"]].head(15).to_string(index=False))
print("... (Full schema saved to schema.csv)")
schema.to_csv(OUT_DIR / "schema.csv", index=False)


# Load into DataFrame and print the shape
df = pd.read_sql_query(f"SELECT * FROM {COMPETITION_TABLE}", con)
print(f"\n DataFrame Shape ")
print(f"df.shape = {df.shape} (should be 7,245 rows)")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]} (9 metadata + 217 labels = should be 226)")
(OUT_DIR / "shape.txt").write_text(
    f"rows: {df.shape[0]}\ncolumns: {df.shape[1]}\n", encoding="utf=8"
)

con.close()
print(f"\nSaved outputs to: {OUT_DIR}")