import sqlite3
from pathlib import Path

# Paths
SOURCE_DB = Path.home() / "Downloads" / "datos_sensores.db"
TARGET_DB = Path("data/greenhouse.db")

# Connect to source and target
src_conn = sqlite3.connect(SOURCE_DB)
tgt_conn = sqlite3.connect(TARGET_DB)

# Enable foreign keys (good habit)
tgt_conn.execute("PRAGMA foreign_keys = ON;")

tables_to_copy = [
    "sensor_data",
    "calendar_executions",
    "schedule",
    "schedule_data"
]

for table in tables_to_copy:
    print(f"\nCopying table: {table}")

    # Read source table
    df = None
    df = src_conn.execute(f"SELECT * FROM {table}").fetchall()

    # Get schema from source
    schema = src_conn.execute(
        f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';"
    ).fetchone()[0]

    tgt_conn.execute(f"DROP TABLE IF EXISTS raw_{table};")
    tgt_conn.execute(f"DROP TABLE IF EXISTS {table};")

    # Create table in target
    tgt_conn.execute(schema)

    # Copy data
    placeholders = ",".join(["?"] * len(df[0]))
    tgt_conn.executemany(
        f"INSERT INTO {table} VALUES ({placeholders})",
        df
    )

    tgt_conn.commit()

    # Validate counts
    count = tgt_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  → {count} rows copied")

# Close connections
src_conn.close()
tgt_conn.close()

print("\nDone.")