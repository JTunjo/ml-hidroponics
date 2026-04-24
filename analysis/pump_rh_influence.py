#!/usr/bin/env python3
"""
Pump execution influence on greenhouse humidity.
Dual analysis: Relative Humidity (RH, %) and Absolute Humidity (AH, g/m³).

AH removes the temperature bias present in RH — warmer air holds more water
vapour so the same pump-added moisture shows a smaller RH rise when it is
hot. AH directly measures how much water the pump added to the air.

Profiles A/B: normalised event response for RH and AH.
Panels  C/D: intensity score and time-of-day grouping, both based on AH.

Sensors (hidroponia_completa.duckdb):
  101 – RH inside        102 – Temperature inside
  103 – RH outside       104 – Temperature outside
  actuator_measure       – pump executions (actuator_id = 1)
"""

import os
import duckdb
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = "data/hidroponia_completa.duckdb"
WIN_BEFORE_MIN = 20
WIN_AFTER_MIN  = 30
RESAMPLE       = "1min"
OUTPUT_DIR     = "analysis"

# Score bin edges — 8 bins → scores 0-7
RH_SCORE_EDGES = [-np.inf, 0, 1.5, 3.0,  5.0,  8.0, 12.0, 16.0, np.inf]  # %
AH_SCORE_EDGES = [-np.inf, 0, 0.15, 0.30, 0.50, 0.80, 1.20, 1.60, np.inf] # g/m³

SLOT_ORDER  = ["06:00–10:00", "10:00–14:00", "14:00–18:00", "18:00–06:00"]
SLOT_COLORS = {
    "06:00–10:00": "#5b9bd5",
    "10:00–14:00": "#ed7d31",
    "14:00–18:00": "#70ad47",
    "18:00–06:00": "#9e5fb5",
}


# ── Physics helpers ───────────────────────────────────────────────────────────
def absolute_humidity(T_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Magnus approximation: RH (%) + T (°C) → AH (g/m³)."""
    e_s = 6.112 * np.exp(17.67 * T_c / (T_c + 243.5))   # saturation vapour pressure (hPa)
    return 216.7 * (rh_pct / 100.0) * e_s / (T_c + 273.15)


# ── Analysis helpers ──────────────────────────────────────────────────────────
def time_slot(ts: pd.Timestamp) -> str:
    h = ts.hour
    if  6 <= h < 10: return "06:00–10:00"
    if 10 <= h < 14: return "10:00–14:00"
    if 14 <= h < 18: return "14:00–18:00"
    return "18:00–06:00"


def assign_score(net_delta: float, edges: list) -> int:
    for score, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if lo < net_delta <= hi:
            return score
    return 7


def lin_slope(s: pd.Series) -> float:
    """OLS slope in signal-units per minute."""
    x = np.arange(len(s))
    return np.polyfit(x, s.values, 1)[0]


def extract_window(series: pd.Series, t: pd.Timestamp) -> np.ndarray:
    """Return a 51-point array centred on the nearest 1-min tick to t."""
    pos = series.index.get_indexer([t], method="nearest")[0]
    ts  = series.index[pos]
    win = pd.date_range(
        ts - pd.Timedelta(minutes=WIN_BEFORE_MIN),
        ts + pd.Timedelta(minutes=WIN_AFTER_MIN),
        freq="1min",
    )
    return series.reindex(win).values


# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_data(db_path: str):
    con = duckdb.connect(db_path, read_only=True)

    execs = con.execute("""
        SELECT
            CAST(start_time AS TIMESTAMP)       AS ts,
            TRY_CAST(execution_time AS INTEGER) AS exec_secs,
            state
        FROM (
            SELECT DISTINCT start_time, execution_time, state
            FROM actuator_measure
            WHERE actuator_id = '1' AND start_time >= '2026-03-02'
        )
        ORDER BY ts
    """).df()
    execs["ts"] = pd.to_datetime(execs["ts"])

    def load_sensor(sid: int, lo: float, hi: float) -> pd.Series:
        df = con.execute(f"""
            SELECT CAST(measure_date AS TIMESTAMP) AS ts,
                   TRY_CAST(data_value AS FLOAT)   AS val
            FROM sensor_data
            WHERE sensor_id = '{sid}'
              AND TRY_CAST(data_value AS FLOAT) BETWEEN {lo} AND {hi}
            ORDER BY ts
        """).df()
        df["ts"] = pd.to_datetime(df["ts"])
        return (df.set_index("ts")["val"]
                  .resample(RESAMPLE).median()
                  .interpolate(limit=5))

    rh_in    = load_sensor(101,  0, 100)   # % RH, inside
    temp_in  = load_sensor(102,  0,  50)   # °C,   inside
    rh_out   = load_sensor(103,  0, 100)   # % RH, outside
    temp_out = load_sensor(104,  0,  50)   # °C,   outside
    con.close()

    # Build AH series on the common 1-min index where both RH and T are available
    def make_ah(rh: pd.Series, temp: pd.Series) -> pd.Series:
        common = pd.concat([rh.rename("rh"), temp.rename("t")], axis=1).dropna()
        return pd.Series(
            absolute_humidity(common["t"].values, common["rh"].values),
            index=common.index
        )

    ah_in  = make_ah(rh_in,  temp_in)
    ah_out = make_ah(rh_out, temp_out)

    return execs, rh_in, rh_out, ah_in, ah_out


# ── 2. Per-event analysis ─────────────────────────────────────────────────────
def analyse_events(execs, rh_in, rh_out, ah_in, ah_out) -> pd.DataFrame:
    records = []

    for _, ev in execs.iterrows():
        t       = ev["ts"]
        pre_s   = t - pd.Timedelta(minutes=WIN_BEFORE_MIN)
        post_e  = t + pd.Timedelta(minutes=WIN_AFTER_MIN)

        # ---------- segment extraction ----------
        rh_in_pre   = rh_in [pre_s : t]
        rh_in_post  = rh_in [t     : post_e]
        rh_out_full = rh_out[pre_s : post_e]
        ah_in_pre   = ah_in [pre_s : t]
        ah_in_post  = ah_in [t     : post_e]
        ah_out_full = ah_out[pre_s : post_e]

        # require 70 % coverage on both signals
        if (len(rh_in_pre)  < WIN_BEFORE_MIN * 0.7 or
                len(rh_in_post) < WIN_AFTER_MIN  * 0.7 or
                len(ah_in_pre)  < WIN_BEFORE_MIN * 0.7 or
                len(ah_in_post) < WIN_AFTER_MIN  * 0.7):
            continue

        # ---------- net Δ vs baselines ----------
        def compute_net(in_pre, in_post, out_full):
            sl_in  = lin_slope(in_pre)
            sl_out = lin_slope(out_full) if len(out_full) > 1 else 0.0
            actual    = in_post.iloc[-1] - in_pre.iloc[-1]
            proj_2a   = sl_in * WIN_AFTER_MIN
            proj_2b   = (0.33 * sl_in + 0.67 * sl_out) * WIN_AFTER_MIN
            return actual, actual - proj_2a, actual - proj_2b

        rh_act, rh_net_2a, rh_net_2b = compute_net(rh_in_pre, rh_in_post, rh_out_full)
        ah_act, ah_net_2a, ah_net_2b = compute_net(ah_in_pre, ah_in_post, ah_out_full)

        records.append({
            "ts":            t,
            "time_slot":     time_slot(t),
            "exec_secs":     ev["exec_secs"],
            # RH metrics
            "rh_pre_mean":   rh_in_pre.mean(),
            "rh_actual":     rh_act,
            "rh_net_2a":     rh_net_2a,
            "rh_net_2b":     rh_net_2b,
            # AH metrics
            "ah_pre_mean":   ah_in_pre.mean(),
            "ah_actual":     ah_act,
            "ah_net_2a":     ah_net_2a,
            "ah_net_2b":     ah_net_2b,
        })

    df = pd.DataFrame(records)
    df["rh_score"] = df["rh_net_2b"].apply(lambda x: assign_score(x, RH_SCORE_EDGES))
    df["ah_score"] = df["ah_net_2b"].apply(lambda x: assign_score(x, AH_SCORE_EDGES))
    df["score"]    = df["ah_score"]   # kept for backward compatibility
    df["time_slot"] = pd.Categorical(df["time_slot"],
                                     categories=SLOT_ORDER, ordered=True)
    return df


# ── 3. Normalised event profiles ──────────────────────────────────────────────
def avg_profile(series_in: pd.Series, series_out: pd.Series,
                ts_list: list) -> tuple:
    """
    Stack normalised 51-point windows (zeroed to mean of 3 min before pump).
    Returns (med_in, std_in, med_out, std_out, n_events).
    """
    mats_in, mats_out = [], []
    for t in ts_list:
        tr_in  = extract_window(series_in,  t).astype(float)
        tr_out = extract_window(series_out, t).astype(float)
        if np.isnan(tr_in).mean() > 0.3 or np.isnan(tr_out).mean() > 0.3:
            continue
        bl_in  = np.nanmean(tr_in [WIN_BEFORE_MIN - 3 : WIN_BEFORE_MIN])
        bl_out = np.nanmean(tr_out[WIN_BEFORE_MIN - 3 : WIN_BEFORE_MIN])
        if np.isnan(bl_in) or np.isnan(bl_out):
            continue
        mats_in.append(tr_in  - bl_in)
        mats_out.append(tr_out - bl_out)
    if not mats_in:
        return None, None, None, None, 0
    mat_in  = np.array(mats_in)
    mat_out = np.array(mats_out)
    return (np.nanmedian(mat_in,  axis=0), np.nanstd(mat_in,  axis=0),
            np.nanmedian(mat_out, axis=0), np.nanstd(mat_out, axis=0),
            len(mat_in))


# ── 4. Summary ────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    w = 72
    print("=" * w)
    print("  PUMP → GREENHOUSE HUMIDITY  |  RH + AH Dual Analysis")
    print("=" * w)
    print(f"  Events : {len(df)}  |  "
          f"{df['ts'].min().date()} → {df['ts'].max().date()}  |  "
          f"Window −{WIN_BEFORE_MIN}/+{WIN_AFTER_MIN} min")
    print()

    for label, col_2a, col_2b, unit in [
        ("Relative Humidity (RH)", "rh_net_2a", "rh_net_2b", "%"),
        ("Absolute Humidity  (AH)", "ah_net_2a", "ah_net_2b", "g/m³"),
    ]:
        print(f"  ── {label} ─────────────────────────────────────────────")
        for bname, col in [("2a inside trend only   ", col_2a),
                            ("2b composite (33/67)   ", col_2b)]:
            t_s, p = stats.ttest_1samp(df[col].dropna(), 0)
            sig    = "  *" if p < 0.05 else ""
            print(f"    Baseline {bname} "
                  f"mean net Δ = {df[col].mean():+.3f} ± {df[col].std():.3f} {unit}"
                  f"  t={t_s:.2f}  p={p:.4f}{sig}")
        print()

    print("  ── AH INTENSITY SCORE (0-7, baseline 2b) ───────────────────────")
    for s in range(8):
        cnt = (df["score"] == s).sum()
        lo, hi = AH_SCORE_EDGES[s], AH_SCORE_EDGES[s + 1]
        if s == 0:
            rng = f"(≤ 0 g/m³)"
        elif hi == np.inf:
            rng = f"(> {lo:.2f} g/m³)"
        else:
            rng = f"({lo:.2f} – {hi:.2f} g/m³)"
        print(f"    Score {s}  {rng:<22}  {cnt:3d}  {'▪' * cnt}")
    print()

    print("  ── BY TIME OF DAY (UTC-5) ───────────────────────────────────────")
    hdr = f"    {'Slot':<16}  {'n':>3}  {'AH score':>9}  {'RH net Δ':>11}  {'AH net Δ':>11}"
    print(hdr)
    print(f"    {'─'*16}  {'─'*3}  {'─'*9}  {'─'*11}  {'─'*11}")
    for slot in SLOT_ORDER:
        sub = df[df["time_slot"] == slot]
        if sub.empty:
            print(f"    {slot:<16}   0")
            continue
        print(f"    {slot:<16}  {len(sub):>3}  "
              f"{sub['score'].mean():>9.2f}  "
              f"{sub['rh_net_2b'].mean():>+10.2f}%  "
              f"{sub['ah_net_2b'].mean():>+9.3f} g/m³")
    print()

    _, p_rh = stats.ttest_1samp(df["rh_net_2b"].dropna(), 0)
    _, p_ah = stats.ttest_1samp(df["ah_net_2b"].dropna(), 0)
    print("  ── VERDICT ──────────────────────────────────────────────────────")
    if p_ah < 0.05 and df["ah_net_2b"].mean() > 0.30:
        print("  AH → RELEVANT: pump adds meaningful water above climate baseline (>0.30 g/m³).")
    elif p_ah < 0.05:
        print("  AH → MARGINAL: statistically significant but small absolute water addition.")
    else:
        print("  AH → NOT CONCLUSIVE: event variance masks the pump signal in the aggregate.")

    if not (p_rh < 0.05) and (p_ah < 0.05):
        print("  NOTE: RH significance is masked by temperature variation — "
              "AH reveals the true effect.")
    elif not (p_rh < 0.05) and not (p_ah < 0.05):
        print("  NOTE: Neither RH nor AH reaches significance — "
              "time-of-day breakdown is the most informative result.")
    print("=" * w)


# ── 5. Plot ───────────────────────────────────────────────────────────────────
def make_plot(df: pd.DataFrame,
              rh_profiles: tuple, ah_profiles: tuple,
              out_path: str) -> None:

    med_rh_in, std_rh_in, med_rh_out, std_rh_out, n_rh = rh_profiles
    med_ah_in, std_ah_in, med_ah_out, std_ah_out, n_ah = ah_profiles
    xmin = np.arange(WIN_BEFORE_MIN + WIN_AFTER_MIN + 1) - WIN_BEFORE_MIN

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Pump Execution Influence on Greenhouse Humidity  —  RH & AH",
        fontsize=13, fontweight="bold"
    )

    # ── A: Normalised RH profile ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.axvline(0, color="dimgray", linestyle="--", linewidth=1.2, label="Pump ON")
    ax.axhline(0, color="black",   linewidth=0.6,  alpha=0.4)
    ax.fill_between(xmin, med_rh_in  - std_rh_in,  med_rh_in  + std_rh_in,
                    alpha=0.18, color="#2a7abf")
    ax.fill_between(xmin, med_rh_out - std_rh_out, med_rh_out + std_rh_out,
                    alpha=0.18, color="#e07030")
    ax.plot(xmin, med_rh_in,  "#2a7abf", linewidth=2, label="Inside RH (101)")
    ax.plot(xmin, med_rh_out, "#e07030", linewidth=2, label="Outside RH (103)")
    ax.set_xlabel("Minutes from pump start")
    ax.set_ylabel("ΔRH from baseline (%)")
    ax.set_title(f"A — Normalised RH profile ± 1 SD  (n={n_rh})\n"
                 "Each trace zeroed to mean RH in the 3 min before pump")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    # ── B: Normalised AH profile ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.axvline(0, color="dimgray", linestyle="--", linewidth=1.2, label="Pump ON")
    ax.axhline(0, color="black",   linewidth=0.6,  alpha=0.4)
    ax.fill_between(xmin, med_ah_in  - std_ah_in,  med_ah_in  + std_ah_in,
                    alpha=0.18, color="#2a7abf")
    ax.fill_between(xmin, med_ah_out - std_ah_out, med_ah_out + std_ah_out,
                    alpha=0.18, color="#e07030")
    ax.plot(xmin, med_ah_in,  "#2a7abf", linewidth=2, label="Inside AH (101+102)")
    ax.plot(xmin, med_ah_out, "#e07030", linewidth=2, label="Outside AH (103+104)")
    ax.set_xlabel("Minutes from pump start")
    ax.set_ylabel("ΔAH from baseline (g/m³)")
    ax.set_title(f"B — Normalised AH profile ± 1 SD  (n={n_ah})\n"
                 "Temperature-corrected water content")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    # ── C: AH score distribution ──────────────────────────────────────────
    ax = axes[1, 0]
    cmap   = plt.cm.RdYlGn
    counts = df["score"].value_counts().reindex(range(8), fill_value=0)
    colors = [cmap(s / 7) for s in counts.index]
    bars   = ax.bar(counts.index, counts.values,
                    color=colors, edgecolor="white", width=0.8)
    ax.bar_label(bars, fmt="%d", padding=2, fontsize=9)
    ax.set_xlabel("AH intensity score  (0 = no effect, 7 = strongest)")
    ax.set_ylabel("Events")
    ax.set_title("C — AH score distribution  (baseline 2b composite)")
    ax.set_xticks(range(8))
    ax.set_ylim(0, counts.max() * 1.22)

    # ── D: AH score by time of day ────────────────────────────────────────
    ax = axes[1, 1]
    slot_series = [df[df["time_slot"] == s]["score"] for s in SLOT_ORDER]
    plot_data   = [s.values for s in slot_series if len(s) > 0]
    plot_labels = [f"{sl}\n(n={len(sd)})"
                   for sl, sd in zip(SLOT_ORDER, slot_series) if len(sd) > 0]
    plot_slots  = [sl for sl, sd in zip(SLOT_ORDER, slot_series) if len(sd) > 0]

    bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for patch, slot in zip(bp["boxes"], plot_slots):
        patch.set_facecolor(SLOT_COLORS[slot])
        patch.set_alpha(0.75)
    for i, d in enumerate(plot_data):
        ax.scatter(i + 1, np.mean(d), color="white", s=55,
                   edgecolors="black", linewidths=1.2, zorder=5,
                   label="Mean" if i == 0 else "")
    ax.set_ylabel("AH intensity score (0-7)")
    ax.set_title("D — AH score by time of day  (UTC-5)")
    ax.set_ylim(-0.5, 7.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved → {out_path}\n")


# ── 6. C+D slide (RH and AH side by side) ────────────────────────────────────
def make_cd_slide(df: pd.DataFrame, out_path: str) -> None:
    """
    2 × 2 slide:
      top row    — score distribution  (C) for RH and AH
      bottom row — score by time slot  (D) for RH and AH, with event counts
    """
    cmap = plt.cm.RdYlGn

    configs = [
        # (score_col, score_edges, unit_label, panel_label)
        ("rh_score", RH_SCORE_EDGES, "%",     "RH"),
        ("ah_score", AH_SCORE_EDGES, "g/m³",  "AH"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Pump Execution — Intensity Score Distribution & Time-of-Day  |  RH vs AH",
        fontsize=13, fontweight="bold"
    )

    for col_idx, (score_col, edges, unit, label) in enumerate(configs):

        # ── C: score distribution ─────────────────────────────────────────
        ax = axes[0, col_idx]
        counts = df[score_col].value_counts().reindex(range(8), fill_value=0)
        colors = [cmap(s / 7) for s in counts.index]
        bars   = ax.bar(counts.index, counts.values,
                        color=colors, edgecolor="white", width=0.8)
        ax.bar_label(bars, fmt="%d", padding=2, fontsize=9)

        # annotate score bin ranges on x-axis
        bin_labels = []
        for s in range(8):
            lo, hi = edges[s], edges[s + 1]
            if s == 0:
                bin_labels.append(f"0\n(≤0)")
            elif hi == np.inf:
                bin_labels.append(f"{s}\n(>{lo:.1f})")
            else:
                bin_labels.append(f"{s}\n({lo:.1f}–{hi:.1f})")
        ax.set_xticks(range(8))
        ax.set_xticklabels(bin_labels, fontsize=7.5)
        ax.set_xlabel(f"Score  [{unit}]", fontsize=9)
        ax.set_ylabel("Events")
        ax.set_title(f"C — {label} score distribution  (n={len(df)}, baseline 2b)")
        ax.set_ylim(0, counts.max() * 1.25)

        # ── D: score by time of day ───────────────────────────────────────
        ax = axes[1, col_idx]
        slot_series = [df[df["time_slot"] == s][score_col] for s in SLOT_ORDER]
        plot_data   = [s.values for s in slot_series if len(s) > 0]
        plot_slots  = [sl for sl, sd in zip(SLOT_ORDER, slot_series) if len(sd) > 0]

        # tick labels: slot name + n count + mean score
        plot_labels = [
            f"{sl}\nn={len(sd)}  (μ={sd.mean():.1f})"
            for sl, sd in zip(SLOT_ORDER, slot_series) if len(sd) > 0
        ]

        bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker="o", markersize=4, alpha=0.5))
        for patch, slot in zip(bp["boxes"], plot_slots):
            patch.set_facecolor(SLOT_COLORS[slot])
            patch.set_alpha(0.75)

        # mean dot + count label inside the box
        for i, (d, sl) in enumerate(zip(plot_data, plot_slots)):
            mean_val = np.mean(d)
            ax.scatter(i + 1, mean_val, color="white", s=60,
                       edgecolors="black", linewidths=1.2, zorder=5,
                       label="Mean" if i == 0 else "")
            ax.text(i + 1, 7.2, f"n={len(d)}",
                    ha="center", va="bottom", fontsize=8, color="dimgray")

        ax.set_ylabel(f"{label} intensity score (0-7)")
        ax.set_title(f"D — {label} score by time of day  (UTC-5)")
        ax.set_ylim(-0.5, 7.8)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"  CD slide saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data …")
    execs, rh_in, rh_out, ah_in, ah_out = load_data(DB_PATH)
    print(f"  {len(execs)} execution records")
    print(f"  RH  — inside: {len(rh_in):,}  outside: {len(rh_out):,} samples")
    print(f"  AH  — inside: {len(ah_in):,}  outside: {len(ah_out):,} samples")

    print("Analysing events …")
    df = analyse_events(execs, rh_in, rh_out, ah_in, ah_out)
    print(f"  {len(df)} events with sufficient data coverage\n")

    print("Building normalised profiles …")
    rh_profiles = avg_profile(rh_in, rh_out, df["ts"].tolist())
    ah_profiles = avg_profile(ah_in, ah_out, df["ts"].tolist())

    print_summary(df)
    make_plot(df, rh_profiles, ah_profiles,
              out_path=f"{OUTPUT_DIR}/pump_rh_influence.png")
    make_cd_slide(df,
                  out_path=f"{OUTPUT_DIR}/pump_cd_slide.png")
