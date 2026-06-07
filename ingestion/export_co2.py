import duckdb
from pathlib import Path

DB_PATH = Path.home() / "Downloads/datos_sensores.db"
OUT_PATH = Path("data/co2_data.csv")

SENSOR_ID = "101"
START_DATE = "2026-05-30 20:00:00"

con = duckdb.connect(":memory:")
con.execute("SET sqlite_all_varchar=true")
con.execute(f"ATTACH '{DB_PATH}' AS src (TYPE SQLITE)")

con.execute(f"""
COPY (
    WITH base AS (
        SELECT
            epoch(time_bucket(INTERVAL '5 minutes', measure_date::TIMESTAMP)) AS bucket_epoch,
            time_bucket(INTERVAL '5 minutes', measure_date::TIMESTAMP) AS bucket,
            TRY_CAST(data_value AS DOUBLE) AS data_value
        FROM src.sensor_data
        WHERE sensor_id = '{SENSOR_ID}'
          AND measure_date >= '{START_DATE}'
          AND TRY_CAST(data_value AS DOUBLE) IS NOT NULL
    ),
    bucketed AS (
        SELECT bucket, bucket_epoch, median(data_value) AS value
        FROM base GROUP BY bucket, bucket_epoch
    ),
    first_bucket AS (SELECT MIN(bucket_epoch) AS t0 FROM bucketed)
    SELECT
        bucket AS measure_date,
        ROUND((bucket_epoch - t0) / 3600.0, 3) AS relative_hour,
        value
    FROM bucketed, first_bucket
    ORDER BY bucket
) TO '{OUT_PATH}' (HEADER, DELIMITER ',')
""")

con.close()

import csv
with open(OUT_PATH) as f:
    rows = sum(1 for _ in f) - 1

print(f"Exported {rows} rows to {OUT_PATH}")
