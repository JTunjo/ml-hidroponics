##READ ME
### Common code
- duckdb data/greenhouse.duckdb < sql_scripts/batch_run.sql
- duckdb data/greenhouse.duckdb < sql_scripts/sp_update_executions.sql
superset run \
  --host 0.0.0.0 \
  --port 8088 \
  --with-threads \
  --reload
superset run -p 8088
- python3 ingestion/02_ingest_weather.py
- python3 ingestion/01_copy_raw_tables.py