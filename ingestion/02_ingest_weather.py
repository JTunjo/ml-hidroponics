import openmeteo_requests
import duckdb
import pandas as pd
import requests_cache
from retry_requests import retry
from pathlib import Path

# ─────────────────────────────
# Open-Meteo client setup
# ─────────────────────────────
cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 4.768071,
    "longitude": -74.133148,
    "start_date": "2026-02-01",
    "end_date": "2026-02-03",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "rain",
        "surface_pressure",
        "wind_speed_10m",
    ],
    "timezone": "auto",
}

responses = openmeteo.weather_api(url, params=params)
response = responses[0]

print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# ─────────────────────────────
# Hourly data extraction
# ─────────────────────────────
hourly = response.Hourly()

hourly_data = {
    "datetime": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ),
    "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
    "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
    "rain": hourly.Variables(2).ValuesAsNumpy(),
    "surface_pressure": hourly.Variables(3).ValuesAsNumpy(),
    "wind_speed_10m": hourly.Variables(4).ValuesAsNumpy(),
}
hourly_df = pd.DataFrame(hourly_data)
DB_PATH = Path("data/greenhouse.duckdb")
con = duckdb.connect(DB_PATH)
con.register("hourly_df", hourly_df)

con.execute("""
INSERT INTO weather_hourly (datetime,temperature_2m,relative_humidity_2m,rain,surface_pressure,wind_speed_10m)
SELECT h.datetime,h.temperature_2m,h.relative_humidity_2m,h.rain,h.surface_pressure,h.wind_speed_10m FROM hourly_df as h LEFT JOIN weather_hourly as wh ON h.datetime = wh.datetime WHERE wh.datetime IS NULL;
""")

con.close()
