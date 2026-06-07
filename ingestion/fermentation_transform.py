import duckdb
import sqlite3

sqlite_conn = sqlite3.connect("data/fermentacion_mayo.db")
duck_conn = duckdb.connect("data/fermentacion_mayo.duckdb")

tables = sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

duck_conn.execute("SET sqlite_all_varchar=true")
duck_conn.execute("ATTACH 'data/fermentacion_mayo.db' AS sqlite_db (TYPE SQLITE)")
for (table,) in tables:
    duck_conn.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM sqlite_db.{table}")
    print(f"Imported: {table}")

duck_conn.close()
sqlite_conn.close()
print("Done.")

# --- CO2 export ---
co2_conn = duckdb.connect(":memory:")
co2_conn.execute("SET sqlite_all_varchar=true")
co2_conn.execute("ATTACH 'data/fermentacion_mayo.db' AS src (TYPE SQLITE)")

co2_conn.execute("""
COPY (
    WITH base AS (
        SELECT
            epoch(time_bucket(INTERVAL '5 minutes', measure_date::TIMESTAMP)) AS bucket_epoch,
            time_bucket(INTERVAL '5 minutes', measure_date::TIMESTAMP) AS bucket,
            TRY_CAST(data_value AS DOUBLE) AS data_value
        FROM src.sensor_data
        WHERE sensor_id = '101'
          AND measure_date >= '2026-05-30 20:00:00'
          AND TRY_CAST(data_value AS DOUBLE) IS NOT NULL
    ),
    bucketed AS (
        SELECT bucket, bucket_epoch, median(data_value) AS value
        FROM base
        GROUP BY bucket, bucket_epoch
    ),
    first_bucket AS (SELECT MIN(bucket_epoch) AS t0 FROM bucketed)
    SELECT
        bucket AS measure_date,
        ROUND((bucket_epoch - t0) / 3600.0, 3) AS relative_hour,
        value
    FROM bucketed, first_bucket
    ORDER BY bucket
) TO 'data/co2_data.csv' (HEADER, DELIMITER ',')
""")

co2_conn.close()
print("Saved data/co2_data.csv")
