import sqlite3
import duckdb
import pandas as pd
from pathlib import Path

# Paths
SOURCE_DB = Path.home() / "Downloads" / "datos_sensores.db"
TARGET_DB = Path("data/greenhouse.duckdb")

# Connect to source (SQLite)
src_conn = sqlite3.connect(SOURCE_DB)

# Connect to target (DuckDB)
tgt_conn = duckdb.connect(TARGET_DB)

tables_to_copy = [
    "sensor_data",
    "calendar_executions",
    "schedule",
    "schedule_data",
]

for table in tables_to_copy:
    print(f"\nCopying table: {table}")

    # Read entire table into pandas
    df = pd.read_sql(f"SELECT * FROM {table}", src_conn)
    df = df.astype(str)
    # Drop + recreate table in DuckDB
    tgt_conn.execute(f"DROP TABLE IF EXISTS {table}")
    seq = f"CREATE TABLE {table} AS SELECT * FROM df"
    print(f"Executing: {seq}")
    tgt_conn.execute(seq)

    # Validate counts
    count = tgt_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  → {count} rows copied")

# Close connections
src_conn.close()
tgt_conn.close()

print("\nDone.")
