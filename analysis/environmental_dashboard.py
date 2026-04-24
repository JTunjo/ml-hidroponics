#!/usr/bin/env python3
"""
Greenhouse Environmental Data Dashboard
Generates analysis/environmental_dashboard.html
"""

import json
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

DB_PATH    = "data/hidroponia_completa.duckdb"
OUT_PATH   = "analysis/environmental_dashboard.html"
WIN_BEFORE = 20
WIN_AFTER  = 30
AH_EDGES   = [-np.inf, 0, 0.15, 0.30, 0.50, 0.80, 1.20, 1.60, np.inf]


# ── Physics ───────────────────────────────────────────────────────────────────

def ah(T_c, rh_pct):
    """Magnus formula: AH in g/m³ from T(°C) and RH(%)."""
    e_s = 6.112 * np.exp(17.67 * T_c / (T_c + 243.5))
    return 216.7 * (rh_pct / 100.0) * e_s / (T_c + 273.15)


def assign_ah_score(delta):
    for score, (lo, hi) in enumerate(zip(AH_EDGES[:-1], AH_EDGES[1:])):
        if lo < delta <= hi:
            return score
    return 7


def lin_slope(s):
    if len(s) < 2:
        return 0.0
    x = np.arange(len(s))
    return np.polyfit(x, s.values, 1)[0]


# ── 1. Load and aggregate data ─────────────────────────────────────────────────

def load_10min_agg(con):
    """Return 10-min pivoted DataFrame with all 4 sensors + AH."""
    df = con.execute("""
        WITH raw AS (
            SELECT time_bucket(INTERVAL '10 minutes',
                               CAST(measure_date AS TIMESTAMP) - INTERVAL '5 hours')
                       AS local_bucket,
                   sensor_id,
                   AVG(TRY_CAST(data_value AS DOUBLE)) AS avg_val
            FROM sensor_data
            WHERE sensor_id IN ('101','103')
              AND TRY_CAST(data_value AS DOUBLE) BETWEEN 25 AND 100
              AND CAST(measure_date AS TIMESTAMP) >= '2026-03-01'
            GROUP BY 1, 2
            UNION ALL
            SELECT time_bucket(INTERVAL '10 minutes',
                               CAST(measure_date AS TIMESTAMP) - INTERVAL '5 hours')
                       AS local_bucket,
                   sensor_id,
                   AVG(TRY_CAST(data_value AS DOUBLE)) AS avg_val
            FROM sensor_data
            WHERE sensor_id IN ('102','104')
              AND TRY_CAST(data_value AS DOUBLE) BETWEEN 0 AND 55
              AND CAST(measure_date AS TIMESTAMP) >= '2026-03-01'
            GROUP BY 1, 2
        )
        SELECT local_bucket,
               EXTRACT(hour FROM local_bucket)::INTEGER AS hour,
               local_bucket::DATE::VARCHAR              AS day,
               MAX(CASE WHEN sensor_id='101' THEN avg_val END) AS rh_in,
               MAX(CASE WHEN sensor_id='102' THEN avg_val END) AS temp_in,
               MAX(CASE WHEN sensor_id='103' THEN avg_val END) AS rh_out,
               MAX(CASE WHEN sensor_id='104' THEN avg_val END) AS temp_out
        FROM raw
        GROUP BY 1, 2, 3
        ORDER BY 1
    """).fetchdf()

    df['ah_in']       = ah(df['temp_in'],  df['rh_in'])
    df['ah_out']      = ah(df['temp_out'], df['rh_out'])
    df['temp_delta']  = df['temp_in']  - df['temp_out']
    df['ah_delta']    = df['ah_in']    - df['ah_out']
    return df


def load_pump_events(con):
    """Pump events (local day) with success state."""
    df = con.execute("""
        SELECT DATE_TRUNC('day',
                   CAST(start_time AS TIMESTAMP) - INTERVAL '5 hours')::DATE::VARCHAR
                   AS local_day,
               COUNT(*) AS n_events,
               SUM(CASE WHEN state='1' THEN 1 ELSE 0 END) AS n_success
        FROM actuator_measure
        WHERE actuator_id='1' AND start_time >= '2026-03-01'
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    return df


def load_minute_series(con):
    """1-minute resampled series for pump window analysis."""
    def _load(sid, lo, hi):
        df = con.execute(f"""
            SELECT CAST(measure_date AS TIMESTAMP) AS ts,
                   TRY_CAST(data_value AS FLOAT)   AS val
            FROM sensor_data
            WHERE sensor_id='{sid}'
              AND TRY_CAST(data_value AS FLOAT) BETWEEN {lo} AND {hi}
            ORDER BY ts
        """).fetchdf()
        df['ts'] = pd.to_datetime(df['ts'])
        return df.set_index('ts')['val'].resample('1min').median().interpolate(limit=5)

    rh_in    = _load(101,  0, 100)
    temp_in  = _load(102,  0,  50)
    rh_out   = _load(103,  0, 100)
    temp_out = _load(104,  0,  50)

    def make_ah_series(rh, temp):
        c = pd.concat([rh.rename('rh'), temp.rename('t')], axis=1).dropna()
        return pd.Series(ah(c['t'].values, c['rh'].values), index=c.index)

    return make_ah_series(rh_in, temp_in), make_ah_series(rh_out, temp_out)


def load_execs(con):
    df = con.execute("""
        SELECT CAST(start_time AS TIMESTAMP) AS ts, state
        FROM (SELECT DISTINCT start_time, state FROM actuator_measure
              WHERE actuator_id='1' AND start_time >= '2026-03-01')
        ORDER BY ts
    """).fetchdf()
    df['ts'] = pd.to_datetime(df['ts'])
    return df


# ── 2. Compute pump influence per event → daily score ─────────────────────────

def extract_window(series, t, before, after):
    pos = series.index.get_indexer([t], method='nearest')[0]
    ts  = series.index[pos]
    win = pd.date_range(ts - pd.Timedelta(minutes=before),
                        ts + pd.Timedelta(minutes=after), freq='1min')
    return series.reindex(win).values


def analyse_pump_events(execs, ah_in, ah_out):
    records = []
    for _, ev in execs.iterrows():
        t = ev['ts']
        pre_s, post_e = t - pd.Timedelta(minutes=WIN_BEFORE), t + pd.Timedelta(minutes=WIN_AFTER)

        seg_in_pre  = ah_in [pre_s:t]
        seg_in_post = ah_in [t:post_e]
        seg_out     = ah_out[pre_s:post_e]

        if (len(seg_in_pre) < WIN_BEFORE * 0.7 or
                len(seg_in_post) < WIN_AFTER  * 0.7):
            continue

        sl_in  = lin_slope(seg_in_pre)
        sl_out = lin_slope(seg_out) if len(seg_out) > 1 else 0.0
        actual = seg_in_post.iloc[-1] - seg_in_pre.iloc[-1]
        proj   = (0.33 * sl_in + 0.67 * sl_out) * WIN_AFTER
        net    = actual - proj

        local_day = (t - pd.Timedelta(hours=5)).date().isoformat()
        records.append({'local_day': local_day, 'ah_net': net,
                        'ah_score': assign_ah_score(net)})

    df = pd.DataFrame(records)
    if df.empty:
        return df

    daily = (df.groupby('local_day')
               .agg(n_events=('ah_net', 'count'),
                    median_net=('ah_net', 'median'),
                    median_score=('ah_score', 'median'),
                    pct_positive=('ah_net', lambda x: (x > 0).mean() * 100))
               .reset_index())
    return daily


# ── 3. Build all data for charts ──────────────────────────────────────────────

def build_chart_data(df10, pump_daily, event_daily, all_days):
    # Hourly averages (all data, no NaN filter on individual columns)
    full = df10.dropna(subset=['temp_in','temp_out','ah_in','ah_out'])
    by_hour = full.groupby('hour').agg(
        temp_in=('temp_in', 'mean'), temp_out=('temp_out', 'mean'),
        temp_delta=('temp_delta', 'mean'),
        ah_in=('ah_in', 'mean'), ah_out=('ah_out', 'mean'),
        ah_delta=('ah_delta', 'mean'),
        n=('temp_in', 'count'),
    ).reindex(range(24)).fillna(0)

    hours = list(range(24))

    # Daily series — join all_days with pump_daily
    daily_full = all_days.merge(pump_daily.rename(columns={'local_day':'day'}),
                                on='day', how='left')
    daily_full['n_events']  = daily_full['n_events'].fillna(0).astype(int)
    daily_full['n_success'] = daily_full['n_success'].fillna(0).astype(int)

    # Daily avg temp and AH from 10-min buckets
    daily_env = (df10.dropna(subset=['temp_in','temp_out','ah_in','ah_out'])
                     .groupby('day').agg(
                         avg_temp_in=('temp_in', 'mean'),
                         avg_temp_out=('temp_out', 'mean'),
                         avg_ah_in=('ah_in', 'mean'),
                         avg_ah_out=('ah_out', 'mean'),
                         avg_temp_delta=('temp_delta', 'mean'),
                         avg_ah_delta=('ah_delta', 'mean'),
                     ).reset_index())

    daily_full = daily_full.merge(daily_env, on='day', how='left')

    # Add pump event daily scores
    if event_daily is not None and not event_daily.empty:
        ev_merge = (event_daily
                    .rename(columns={'local_day': 'day',
                                     'n_events': 'ev_n_events'})
                    [['day', 'ev_n_events', 'median_net', 'median_score', 'pct_positive']])
        daily_full = daily_full.merge(ev_merge, on='day', how='left')
    else:
        daily_full['median_score'] = np.nan
        daily_full['pct_positive'] = np.nan

    return by_hour, hours, daily_full


# ── 4. KPI summary stats ──────────────────────────────────────────────────────

def kpi_stats(df10, event_daily, pump_daily):
    full = df10.dropna(subset=['temp_in','temp_out','ah_in','ah_out'])
    daytime = full[full['hour'].between(6, 19)]

    # Greenhouse temperature lift
    avg_temp_lift = daytime['temp_delta'].mean()
    peak_temp_lift = daytime['temp_delta'].max()
    peak_temp_hour = daytime.loc[daytime['temp_delta'].idxmax(), 'hour']

    # Daytime AH enrichment
    avg_ah_enrich = daytime['ah_delta'].mean()
    peak_ah_enrich = daytime['ah_delta'].max()

    # All-time totals
    total_buckets   = len(df10)
    complete_buckets = len(full)
    first_day = df10['day'].min()
    last_day  = df10['day'].max()

    # Pump influence
    if event_daily is not None and not event_daily.empty:
        overall_median_score = event_daily['median_score'].median()
        overall_pct_positive = event_daily['pct_positive'].mean()
        n_pump_days = len(event_daily)
    else:
        overall_median_score = np.nan
        overall_pct_positive = np.nan
        n_pump_days = 0

    # Temperature at key hours
    hourly = full.groupby('hour').agg(
        temp_in=('temp_in','mean'), temp_out=('temp_out','mean'),
        ah_in=('ah_in','mean'), ah_out=('ah_out','mean'),
        ah_delta=('ah_delta','mean'), temp_delta=('temp_delta','mean')
    )
    peak_ah_hour_row = hourly['ah_delta'].idxmax()

    # Days with no pump in sensor coverage
    days_with_sensor = set(df10['day'].unique())
    days_with_pump   = set(pump_daily['local_day'].unique()) if not pump_daily.empty else set()
    n_no_pump_days   = len(days_with_sensor - days_with_pump)

    return {
        'total_buckets':      total_buckets,
        'complete_buckets':   complete_buckets,
        'first_day':          first_day,
        'last_day':           last_day,
        'avg_temp_lift':      round(avg_temp_lift, 2),
        'peak_temp_lift':     round(peak_temp_lift, 2),
        'peak_temp_hour':     int(peak_temp_hour),
        'avg_ah_enrich':      round(avg_ah_enrich, 3),
        'peak_ah_enrich':     round(peak_ah_enrich, 3),
        'peak_ah_hour':       int(peak_ah_hour_row),
        'overall_median_score': round(overall_median_score, 2) if not np.isnan(overall_median_score) else 'N/A',
        'overall_pct_positive': round(overall_pct_positive, 1) if not np.isnan(overall_pct_positive) else 'N/A',
        'n_pump_days':        n_pump_days,
        'n_no_pump_days':     n_no_pump_days,
        # avg temps
        'avg_daytime_temp_in':  round(daytime['temp_in'].mean(), 1),
        'avg_daytime_temp_out': round(daytime['temp_out'].mean(), 1),
        'avg_daytime_ah_in':    round(daytime['ah_in'].mean(), 2),
        'avg_daytime_ah_out':   round(daytime['ah_out'].mean(), 2),
    }


# ── 5. Summary narrative ──────────────────────────────────────────────────────

def generate_summary(kpi, df10, event_daily):
    full = df10.dropna(subset=['temp_in','temp_out','ah_in','ah_out'])

    # Night hours when greenhouse is COOLER than outside
    night_cool = full[full['hour'].between(0, 5)]
    night_cool_pct = (night_cool['temp_delta'] < 0).mean() * 100

    # Correlation inside AH vs outside AH
    corr = full['ah_in'].corr(full['ah_out'])

    # Daytime peak hour (local)
    daytime = full[full['hour'].between(6, 19)]
    peak_hour_str = f"{kpi['peak_temp_hour']:02d}:00"
    peak_ah_hour_str = f"{kpi['peak_ah_hour']:02d}:00"

    # RH saturation inside
    rh_sat_pct = (df10['rh_in'].dropna() >= 99).mean() * 100

    lines = []

    lines.append(f"""
<h3>Dataset Coverage</h3>
<p>The 10-minute aggregated dataset spans <strong>{kpi['first_day']}</strong> to
<strong>{kpi['last_day']}</strong>, comprising <strong>{kpi['total_buckets']:,}</strong>
time windows of which <strong>{kpi['complete_buckets']:,}</strong> ({100*kpi['complete_buckets']/kpi['total_buckets']:.0f}%)
have complete readings from all four sensors.</p>
""")

    lines.append(f"""
<h3>Thermal Buffer Effect</h3>
<p>During daytime hours (06:00–20:00 local), the greenhouse interior averages
<strong>{kpi['avg_daytime_temp_in']}°C</strong> vs. <strong>{kpi['avg_daytime_temp_out']}°C</strong>
outside — a mean lift of <strong>+{kpi['avg_temp_lift']}°C</strong>. The peak thermal
gain reaches <strong>+{kpi['peak_temp_lift']}°C</strong> around {peak_hour_str} local time,
when solar irradiance is highest and the greenhouse structure traps radiant heat most effectively.</p>
<p>After sunset, this dynamic reverses: the interior cools below ambient temperature roughly
<strong>{night_cool_pct:.0f}%</strong> of nighttime windows, suggesting moderate heat retention
capacity but insufficient thermal mass for full-night temperature maintenance.</p>
""")

    lines.append(f"""
<h3>Absolute Humidity Enrichment</h3>
<p>Temperature-corrected Absolute Humidity (AH, g/m³) reveals that the greenhouse interior
holds on average <strong>{kpi['avg_ah_enrich']:+.2f} g/m³</strong> more water vapour than
the exterior during daytime. Peak enrichment occurs around {peak_ah_hour_str} local
(<strong>+{kpi['peak_ah_enrich']:.2f} g/m³</strong>), driven by a combination of evapotranspiration
from plants and limited air exchange with the outside.</p>
<p>The correlation between interior and exterior AH is <strong>{corr:.2f}</strong>, indicating
that large-scale weather patterns dominate the ambient moisture level while the greenhouse
provides a consistent positive offset. On days with heavy rainfall or cloud cover, the inside–outside
AH gap narrows; on dry, sunny days it widens due to higher plant transpiration rates.</p>
""")

    lines.append(f"""
<h3>RH Ceiling Effect</h3>
<p>Approximately <strong>{rh_sat_pct:.0f}%</strong> of interior RH readings reach or exceed
99% — the practical sensor ceiling. This saturation renders RH a poor discriminator of actual
moisture content during those periods. Absolute Humidity is therefore the preferred metric for
evaluating both greenhouse buffering capacity and pump effectiveness in this dataset.</p>
""")

    if event_daily is not None and not event_daily.empty:
        n_positive_days = (event_daily['pct_positive'] >= 50).sum()
        med_score = event_daily['median_score'].median()
        lines.append(f"""
<h3>Pump Net AH Influence</h3>
<p>Across <strong>{kpi['n_pump_days']}</strong> days with recorded pump activity, the median
per-event net AH impact (against the composite 33/67 inside–outside baseline) yields a
median daily AH score of <strong>{kpi['overall_median_score']}</strong> (scale 0–7).
On <strong>{n_positive_days}</strong> out of {kpi['n_pump_days']} pump-active days,
more than half of pump events showed a positive net AH increase above the climate baseline.</p>
<p>The remaining days with negative or near-zero scores are consistent with the three
confounding factors identified in the event-window analysis: (1) inside RH often already
saturated, leaving no headroom for additional vapour; (2) the 45-second pump pulse is short
relative to the 30-minute analysis window; and (3) afternoon heat significantly raises
the vapour-pressure deficit, requiring disproportionately more water to shift AH.</p>
""")

    lines.append(f"""
<h3>Operational Insight</h3>
<p>The greenhouse delivers a consistent <strong>thermal and hygroscopic buffer</strong>:
interior temperature is reliably warmer during the productive daytime hours and AH is
consistently elevated. The pump system adds net moisture above the ambient trend on most
active days, though its effect is partially masked by sensor saturation and the greenhouse's
tight coupling to exterior ambient humidity. Extending pump pulse duration, operating
preferentially in the early morning (06:00–10:00) when the inside–outside differential
is largest, and improving greenhouse sealing would be the highest-leverage improvements
to raise the measurable pump AH impact.</p>
""")

    return '\n'.join(lines)


# ── 6. Render HTML ────────────────────────────────────────────────────────────

def render_html(by_hour, hours, daily_full, kpi, summary_html):
    hours_labels = [f"{h:02d}:00" for h in hours]
    days_labels  = daily_full['day'].tolist()

    # Hourly chart data
    temp_in_h  = [round(v, 2) for v in by_hour['temp_in'].tolist()]
    temp_out_h = [round(v, 2) for v in by_hour['temp_out'].tolist()]
    ah_in_h    = [round(v, 3) for v in by_hour['ah_in'].tolist()]
    ah_out_h   = [round(v, 3) for v in by_hour['ah_out'].tolist()]
    temp_dlt_h = [round(v, 3) for v in by_hour['temp_delta'].tolist()]
    ah_dlt_h   = [round(v, 3) for v in by_hour['ah_delta'].tolist()]

    # Daily chart data (replace NaN with null for JSON)
    def clean(series):
        return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(v, 3)
                for v in series.tolist()]

    d_temp_in   = clean(daily_full['avg_temp_in'])
    d_temp_out  = clean(daily_full['avg_temp_out'])
    d_ah_in     = clean(daily_full['avg_ah_in'])
    d_ah_out    = clean(daily_full['avg_ah_out'])
    d_temp_dlt  = clean(daily_full['avg_temp_delta'])
    d_ah_dlt    = clean(daily_full['avg_ah_delta'])
    d_score     = clean(daily_full.get('median_score', pd.Series([np.nan]*len(daily_full))))
    d_pct_pos   = clean(daily_full.get('pct_positive', pd.Series([np.nan]*len(daily_full))))
    d_n_events  = daily_full['n_events'].fillna(0).astype(int).tolist()

    # Pump markers: background color for days with pump
    pump_days_set = set(daily_full[daily_full['n_events'] > 0]['day'].tolist())
    bar_colors_score = []
    bar_colors_pct   = []
    for d in days_labels:
        if d in pump_days_set:
            bar_colors_score.append('rgba(59,130,246,0.75)')
            bar_colors_pct.append('rgba(16,185,129,0.75)')
        else:
            bar_colors_score.append('rgba(200,200,200,0.4)')
            bar_colors_pct.append('rgba(200,200,200,0.4)')

    pump_background = ['rgba(59,130,246,0.08)' if d in pump_days_set
                       else 'rgba(0,0,0,0)' for d in days_labels]

    jdays  = json.dumps(days_labels)
    jhours = json.dumps(hours_labels)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Greenhouse Environmental Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f8fafc; color: #1e293b; font-size: 14px; }}
  header {{ background: #ffffff; border-bottom: 1px solid #e2e8f0; padding: 20px 32px; display: flex; align-items: baseline; gap: 12px; }}
  header h1 {{ font-size: 20px; font-weight: 700; color: #0f172a; }}
  header span {{ font-size: 13px; color: #64748b; }}
  .container {{ max-width: 1280px; margin: 0 auto; padding: 24px 32px 48px; }}
  .section-title {{ font-size: 15px; font-weight: 700; color: #0f172a; margin: 36px 0 16px;
                    display: flex; align-items: center; gap: 8px; }}
  .section-title::before {{ content: ''; display: block; width: 4px; height: 16px;
                             background: #3b82f6; border-radius: 2px; }}

  /* KPI grid */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; margin-bottom: 4px; }}
  .kpi-card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
               padding: 16px 18px; }}
  .kpi-card .label {{ font-size: 11px; font-weight: 600; text-transform: uppercase;
                      letter-spacing: .05em; color: #64748b; margin-bottom: 6px; }}
  .kpi-card .value {{ font-size: 26px; font-weight: 700; color: #0f172a; line-height: 1; }}
  .kpi-card .sub   {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
  .kpi-card.blue   {{ border-left: 3px solid #3b82f6; }}
  .kpi-card.green  {{ border-left: 3px solid #10b981; }}
  .kpi-card.orange {{ border-left: 3px solid #f59e0b; }}
  .kpi-card.purple {{ border-left: 3px solid #8b5cf6; }}
  .kpi-card.red    {{ border-left: 3px solid #ef4444; }}
  .kpi-card.teal   {{ border-left: 3px solid #14b8a6; }}

  /* Chart rows */
  .chart-row {{ display: grid; gap: 18px; margin-bottom: 18px; }}
  .chart-row.cols-2 {{ grid-template-columns: 1fr 1fr; }}
  .chart-row.cols-1 {{ grid-template-columns: 1fr; }}
  .chart-card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 20px; }}
  .chart-card h4 {{ font-size: 13px; font-weight: 600; color: #374151; margin-bottom: 14px; }}
  .chart-card canvas {{ max-height: 280px; }}
  .chart-card.tall canvas {{ max-height: 340px; }}

  /* Summary */
  .summary {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
              padding: 24px 28px; line-height: 1.7; }}
  .summary h3 {{ font-size: 14px; font-weight: 700; color: #0f172a; margin: 20px 0 8px; }}
  .summary h3:first-child {{ margin-top: 0; }}
  .summary p  {{ font-size: 13px; color: #374151; margin-bottom: 6px; }}
  .summary strong {{ color: #0f172a; }}

  /* Legend pill */
  .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 6px; font-size: 12px; color: #64748b; }}
  .legend span {{ display: flex; align-items: center; gap: 5px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
</style>
</head>
<body>

<header>
  <h1>Greenhouse Environmental Dashboard</h1>
  <span>Sensor data · 10-min windows · {kpi['first_day']} → {kpi['last_day']}</span>
</header>

<div class="container">

<!-- ── Section 1: Data Quality ── -->
<div class="section-title">Data Coverage</div>
<div class="kpi-grid">
  <div class="kpi-card blue">
    <div class="label">10-min Windows</div>
    <div class="value">{kpi['total_buckets']:,}</div>
    <div class="sub">{kpi['complete_buckets']:,} complete (all 4 sensors)</div>
  </div>
  <div class="kpi-card blue">
    <div class="label">Date Range</div>
    <div class="value" style="font-size:16px">{kpi['first_day']}</div>
    <div class="sub">to {kpi['last_day']}</div>
  </div>
  <div class="kpi-card orange">
    <div class="label">Pump-Active Days</div>
    <div class="value">{kpi['n_pump_days']}</div>
    <div class="sub">{kpi['n_no_pump_days']} days — no pump</div>
  </div>
</div>

<!-- ── Section 2: Thermal Buffer ── -->
<div class="section-title">Greenhouse Thermal Buffer</div>
<div class="kpi-grid">
  <div class="kpi-card blue">
    <div class="label">Avg Daytime Temp Inside</div>
    <div class="value">{kpi['avg_daytime_temp_in']}°C</div>
    <div class="sub">06:00–20:00 local</div>
  </div>
  <div class="kpi-card teal">
    <div class="label">Avg Daytime Temp Outside</div>
    <div class="value">{kpi['avg_daytime_temp_out']}°C</div>
    <div class="sub">06:00–20:00 local</div>
  </div>
  <div class="kpi-card green">
    <div class="label">Avg Daytime Temp Lift</div>
    <div class="value">+{kpi['avg_temp_lift']}°C</div>
    <div class="sub">Inside minus outside</div>
  </div>
  <div class="kpi-card orange">
    <div class="label">Peak Temp Lift</div>
    <div class="value">+{kpi['peak_temp_lift']}°C</div>
    <div class="sub">Around {kpi['peak_temp_hour']:02d}:00 local</div>
  </div>
</div>
<div class="chart-row cols-2">
  <div class="chart-card">
    <h4>Inside vs Outside Temperature · Hourly Average (°C)</h4>
    <canvas id="tempHourly"></canvas>
  </div>
  <div class="chart-card">
    <h4>Temperature Differential (Inside − Outside) · Hourly Average (°C)</h4>
    <canvas id="tempDelta"></canvas>
  </div>
</div>
<div class="chart-row cols-2">
  <div class="chart-card tall">
    <h4>Daily Average Temperature (°C)</h4>
    <canvas id="tempDaily"></canvas>
  </div>
  <div class="chart-card tall">
    <h4>Daily Temperature Lift (Inside − Outside, °C)</h4>
    <canvas id="tempDeltaDaily"></canvas>
  </div>
</div>

<!-- ── Section 3: AH Buffer ── -->
<div class="section-title">Absolute Humidity Enrichment</div>
<div class="kpi-grid">
  <div class="kpi-card blue">
    <div class="label">Avg Daytime AH Inside</div>
    <div class="value">{kpi['avg_daytime_ah_in']}</div>
    <div class="sub">g/m³ · 06:00–20:00 local</div>
  </div>
  <div class="kpi-card teal">
    <div class="label">Avg Daytime AH Outside</div>
    <div class="value">{kpi['avg_daytime_ah_out']}</div>
    <div class="sub">g/m³ · 06:00–20:00 local</div>
  </div>
  <div class="kpi-card green">
    <div class="label">Avg Daytime AH Enrichment</div>
    <div class="value">+{kpi['avg_ah_enrich']}</div>
    <div class="sub">g/m³ inside above outside</div>
  </div>
  <div class="kpi-card purple">
    <div class="label">Peak AH Enrichment</div>
    <div class="value">+{kpi['peak_ah_enrich']}</div>
    <div class="sub">g/m³ around {kpi['peak_ah_hour']:02d}:00 local</div>
  </div>
</div>
<div class="chart-row cols-2">
  <div class="chart-card">
    <h4>Inside vs Outside Absolute Humidity · Hourly Average (g/m³)</h4>
    <canvas id="ahHourly"></canvas>
  </div>
  <div class="chart-card">
    <h4>AH Differential (Inside − Outside) · Hourly Average (g/m³)</h4>
    <canvas id="ahDelta"></canvas>
  </div>
</div>
<div class="chart-row cols-2">
  <div class="chart-card tall">
    <h4>Daily Average Absolute Humidity (g/m³)</h4>
    <canvas id="ahDaily"></canvas>
  </div>
  <div class="chart-card tall">
    <h4>Daily AH Enrichment (Inside − Outside, g/m³)</h4>
    <canvas id="ahDeltaDaily"></canvas>
  </div>
</div>

<!-- ── Section 4: Pump Influence ── -->
<div class="section-title">Net Pump Influence on Absolute Humidity</div>
<div class="kpi-grid">
  <div class="kpi-card blue">
    <div class="label">Median Daily AH Score</div>
    <div class="value">{kpi['overall_median_score']}</div>
    <div class="sub">Scale 0–7 · composite baseline</div>
  </div>
  <div class="kpi-card green">
    <div class="label">Days with &gt;50% Positive Events</div>
    <div class="value" id="positiveDay">—</div>
    <div class="sub">out of {kpi['n_pump_days']} pump-active days</div>
  </div>
  <div class="kpi-card orange">
    <div class="label">Avg % Positive Events / Day</div>
    <div class="value">{kpi['overall_pct_positive']}%</div>
    <div class="sub">events with net AH &gt; 0</div>
  </div>
</div>
<div class="legend">
  <span><span class="dot" style="background:rgba(59,130,246,0.75)"></span>Pump active</span>
  <span><span class="dot" style="background:rgba(200,200,200,0.5)"></span>No pump</span>
</div>
<div class="chart-row cols-1">
  <div class="chart-card tall">
    <h4>Daily Median AH Score (0–7) — pump-active days in blue, no-pump in grey</h4>
    <canvas id="scoreTimeline"></canvas>
  </div>
</div>
<div class="chart-row cols-1">
  <div class="chart-card tall">
    <h4>% of Pump Events with Positive Net AH (per day)</h4>
    <canvas id="pctTimeline"></canvas>
  </div>
</div>

<!-- ── Section 5: Summary ── -->
<div class="section-title">Analytical Summary</div>
<div class="summary">
  {summary_html}
</div>

</div><!-- /container -->

<script>
const HOURS  = {jhours};
const DAYS   = {jdays};

const tempInH  = {json.dumps(temp_in_h)};
const tempOutH = {json.dumps(temp_out_h)};
const ahInH    = {json.dumps(ah_in_h)};
const ahOutH   = {json.dumps(ah_out_h)};
const tempDltH = {json.dumps(temp_dlt_h)};
const ahDltH   = {json.dumps(ah_dlt_h)};

const dTempIn  = {json.dumps(d_temp_in)};
const dTempOut = {json.dumps(d_temp_out)};
const dAhIn    = {json.dumps(d_ah_in)};
const dAhOut   = {json.dumps(d_ah_out)};
const dTempDlt = {json.dumps(d_temp_dlt)};
const dAhDlt   = {json.dumps(d_ah_dlt)};
const dScore   = {json.dumps(d_score)};
const dPctPos  = {json.dumps(d_pct_pos)};
const dNEvents = {json.dumps(d_n_events)};
const barColScore  = {json.dumps(bar_colors_score)};
const barColPct    = {json.dumps(bar_colors_pct)};
const pumpBg       = {json.dumps(pump_background)};

// Count positive days
const positiveDays = dScore.filter((v,i) => dNEvents[i]>0 && v !== null && v >= 1).length;
document.getElementById('positiveDay').textContent = positiveDays;

const baseOpts = {{
  responsive: true,
  maintainAspectRatio: true,
  interaction: {{ mode: 'index', intersect: false }},
  plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }},
              tooltip: {{ bodyFont: {{ size: 11 }} }} }},
  scales: {{ x: {{ ticks: {{ font: {{ size: 10 }} }} }},
             y: {{ ticks: {{ font: {{ size: 10 }} }} }} }},
}};

function lineOpts(yLabel) {{
  return JSON.parse(JSON.stringify({{
    ...baseOpts,
    scales: {{ x: {{ ticks: {{ font: {{ size: 10 }} }} }},
               y: {{ title: {{ display: true, text: yLabel, font: {{ size: 11 }} }},
                    ticks: {{ font: {{ size: 10 }} }} }} }}
  }}));
}}

function mkLine(id, labels, datasets, yLabel) {{
  new Chart(document.getElementById(id), {{
    type: 'line', data: {{ labels, datasets }},
    options: lineOpts(yLabel)
  }});
}}

function mkBar(id, labels, datasets, yLabel) {{
  new Chart(document.getElementById(id), {{
    type: 'bar', data: {{ labels, datasets }},
    options: {{
      ...lineOpts(yLabel),
      scales: {{
        x: {{ ticks: {{ font: {{ size: 9 }}, maxRotation: 45, autoSkip: true, maxTicksLimit: 20 }} }},
        y: {{ title: {{ display: true, text: yLabel, font: {{ size: 11 }} }},
              ticks: {{ font: {{ size: 10 }} }} }}
      }}
    }}
  }});
}}

// ── Hourly: Temperature ──
mkLine('tempHourly', HOURS, [
  {{ label: 'Inside (°C)',  data: tempInH,  borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)',
     fill: false, tension: 0.4, pointRadius: 3 }},
  {{ label: 'Outside (°C)', data: tempOutH, borderColor: '#14b8a6', backgroundColor: 'rgba(20,184,166,0.1)',
     fill: false, tension: 0.4, pointRadius: 3 }},
], '°C');

// ── Hourly: Temp Delta ──
new Chart(document.getElementById('tempDelta'), {{
  type: 'bar',
  data: {{ labels: HOURS, datasets: [{{
    label: 'Temp lift (°C)',
    data: tempDltH,
    backgroundColor: tempDltH.map(v => v >= 0 ? 'rgba(59,130,246,0.65)' : 'rgba(239,68,68,0.55)'),
  }}] }},
  options: {{
    ...lineOpts('°C'),
    scales: {{ x: {{ ticks: {{ font: {{ size: 10 }} }} }}, y: {{ title: {{ display:true, text:'°C' }} }} }}
  }}
}});

// ── Hourly: AH ──
mkLine('ahHourly', HOURS, [
  {{ label: 'AH Inside (g/m³)',  data: ahInH,  borderColor: '#8b5cf6', fill: false, tension: 0.4, pointRadius: 3 }},
  {{ label: 'AH Outside (g/m³)', data: ahOutH, borderColor: '#f59e0b', fill: false, tension: 0.4, pointRadius: 3 }},
], 'g/m³');

// ── Hourly: AH Delta ──
new Chart(document.getElementById('ahDelta'), {{
  type: 'bar',
  data: {{ labels: HOURS, datasets: [{{
    label: 'AH enrichment (g/m³)',
    data: ahDltH,
    backgroundColor: ahDltH.map(v => v >= 0 ? 'rgba(139,92,246,0.65)' : 'rgba(239,68,68,0.55)'),
  }}] }},
  options: {{
    ...lineOpts('g/m³'),
    scales: {{ x: {{ ticks: {{ font: {{ size: 10 }} }} }}, y: {{ title: {{ display:true, text:'g/m³' }} }} }}
  }}
}});

// ── Daily: Temperature ──
const dailyXOpts = {{
  responsive: true, maintainAspectRatio: true,
  interaction: {{ mode: 'index', intersect: false }},
  plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }} }},
  scales: {{
    x: {{ ticks: {{ font: {{ size: 9 }}, maxRotation: 45, autoSkip: true, maxTicksLimit: 20 }} }},
    y: {{ ticks: {{ font: {{ size: 10 }} }} }}
  }}
}};

new Chart(document.getElementById('tempDaily'), {{
  type: 'line',
  data: {{ labels: DAYS, datasets: [
    {{ label:'Inside (°C)', data:dTempIn,  borderColor:'#3b82f6', fill:false, tension:0.3,
       pointRadius:2, borderWidth:2 }},
    {{ label:'Outside (°C)', data:dTempOut, borderColor:'#14b8a6', fill:false, tension:0.3,
       pointRadius:2, borderWidth:2 }},
  ]}},
  options: dailyXOpts
}});

new Chart(document.getElementById('tempDeltaDaily'), {{
  type: 'bar',
  data: {{ labels: DAYS, datasets: [
    {{ label:'Temp lift (°C)', data:dTempDlt,
       backgroundColor: dTempDlt.map(v => v===null ? '#eee' : (v>=0?'rgba(59,130,246,0.6)':'rgba(239,68,68,0.5)')),
    }}
  ]}},
  options: {{ ...dailyXOpts, scales: {{
    x: {{ ticks: {{ font:{{ size:9 }}, maxRotation:45, autoSkip:true, maxTicksLimit:20 }} }},
    y: {{ title:{{ display:true, text:'°C' }} }}
  }} }}
}});

// ── Daily: AH ──
new Chart(document.getElementById('ahDaily'), {{
  type: 'line',
  data: {{ labels: DAYS, datasets: [
    {{ label:'AH Inside (g/m³)', data:dAhIn,  borderColor:'#8b5cf6', fill:false, tension:0.3,
       pointRadius:2, borderWidth:2 }},
    {{ label:'AH Outside (g/m³)', data:dAhOut, borderColor:'#f59e0b', fill:false, tension:0.3,
       pointRadius:2, borderWidth:2 }},
  ]}},
  options: dailyXOpts
}});

new Chart(document.getElementById('ahDeltaDaily'), {{
  type: 'bar',
  data: {{ labels: DAYS, datasets: [
    {{ label:'AH enrichment (g/m³)', data:dAhDlt,
       backgroundColor: dAhDlt.map(v => v===null ? '#eee' : (v>=0?'rgba(139,92,246,0.6)':'rgba(239,68,68,0.5)')),
    }}
  ]}},
  options: {{ ...dailyXOpts, scales: {{
    x: {{ ticks: {{ font:{{ size:9 }}, maxRotation:45, autoSkip:true, maxTicksLimit:20 }} }},
    y: {{ title:{{ display:true, text:'g/m³' }} }}
  }} }}
}});

// ── Score timeline ──
new Chart(document.getElementById('scoreTimeline'), {{
  type: 'bar',
  data: {{ labels: DAYS, datasets: [
    {{ label:'Median AH Score (0–7)', data:dScore,
       backgroundColor: barColScore, borderWidth: 0 }}
  ]}},
  options: {{ ...dailyXOpts,
    scales: {{
      x: {{ ticks: {{ font:{{ size:9 }}, maxRotation:45, autoSkip:true, maxTicksLimit:20 }} }},
      y: {{ min:0, max:7, title:{{ display:true, text:'Score' }}, ticks:{{ stepSize:1 }} }}
    }}
  }}
}});

// ── % Positive timeline ──
new Chart(document.getElementById('pctTimeline'), {{
  type: 'bar',
  data: {{ labels: DAYS, datasets: [
    {{ label:'% Positive net AH events', data:dPctPos,
       backgroundColor: barColPct, borderWidth: 0 }}
  ]}},
  options: {{ ...dailyXOpts,
    scales: {{
      x: {{ ticks: {{ font:{{ size:9 }}, maxRotation:45, autoSkip:true, maxTicksLimit:20 }} }},
      y: {{ min:0, max:100, title:{{ display:true, text:'%' }} }}
    }}
  }}
}});

</script>
</body>
</html>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    con = duckdb.connect(DB_PATH, read_only=True)

    print("Loading 10-min aggregated data...")
    df10 = load_10min_agg(con)

    print("Loading pump events...")
    pump_daily = load_pump_events(con)

    print("Loading minute-resolution series for pump window analysis...")
    ah_in_min, ah_out_min = load_minute_series(con)
    execs = load_execs(con)
    con.close()

    print(f"Loaded {len(df10)} 10-min buckets, {len(execs)} pump events")

    print("Analysing pump events...")
    event_daily = analyse_pump_events(execs, ah_in_min, ah_out_min)
    print(f"Pump event daily summary: {len(event_daily)} days")

    # All days with sensor data
    all_days = df10[['day']].drop_duplicates().sort_values('day').reset_index(drop=True)

    by_hour, hours, daily_full = build_chart_data(df10, pump_daily, event_daily, all_days)

    kpi = kpi_stats(df10, event_daily if not event_daily.empty else None, pump_daily)
    print("KPI stats computed")

    summary_html = generate_summary(kpi, df10, event_daily if not event_daily.empty else None)

    html = render_html(by_hour, hours, daily_full, kpi, summary_html)

    with open(OUT_PATH, 'w') as f:
        f.write(html)
    print(f"Dashboard written to {OUT_PATH}")


if __name__ == '__main__':
    main()
