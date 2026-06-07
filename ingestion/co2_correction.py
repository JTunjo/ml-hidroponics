import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# ─── Models ──────────────────────────────────────────────────────────────────

def fit_rising(x, y, degree=2):
    """
    Polynomial fit to the rising window before saturation.
    Returns (callable, coefficients).
    """
    coeffs = np.polyfit(x, y, deg=degree)
    return np.poly1d(coeffs), coeffs


def _exp_decay(t, a, b, c):
    return a * np.exp(b * t) + c


def fit_dropping(x, y):
    """
    Exponential decay fit to the first drop window after saturation.
    y = a * exp(b * (x - x0)) + c,  b < 0
    Returns (callable, (a, b, c), x0).
    """
    x0 = x[0]
    t = x - x0
    # Initial guesses from data shape
    c0 = float(y[-1])
    a0 = float(y[0]) - c0
    dt = float(t[1]) if len(t) > 1 else 1.0
    ratio = (float(y[1]) - c0) / a0 if a0 != 0 else 0.5
    b0 = np.log(max(ratio, 1e-6)) / dt if dt != 0 else -2.0

    popt, _ = curve_fit(_exp_decay, t, y, p0=[a0, b0, c0], maxfev=20_000)

    def fn(x_val):
        return _exp_decay(np.asarray(x_val, dtype=float) - x0, *popt)

    return fn, tuple(popt), x0


# ─── Gap filler ──────────────────────────────────────────────────────────────

def fill_anomaly(df, plateau_start, first_drop, n_rising=180, n_drop=9, peak_offset=2):
    """
    Synthesise corrected_value for every saturated row in the plateau.

    plateau_start : pandas iloc index of the first saturated (==40 000) row
    first_drop    : pandas iloc index of the first real sub-40 000 row
    peak_offset   : steps before first_drop where the two curves cross
    """
    # Rising fit: n_rising points immediately before the plateau
    rise_slice = df.iloc[plateau_start - n_rising : plateau_start]
    rise_fn, rise_coeffs = fit_rising(
        rise_slice["relative_hour"].values,
        rise_slice["value"].values,
    )

    # Drop fit: n_drop points starting at first_drop
    drop_slice = df.iloc[first_drop : first_drop + n_drop]
    drop_fn, drop_popt, drop_x0 = fit_dropping(
        drop_slice["relative_hour"].values,
        drop_slice["value"].values,
    )

    peak_idx = first_drop - peak_offset  # absolute pandas index

    col = df.columns.get_loc("corrected_value")
    rh_col = df.columns.get_loc("relative_hour")

    for i in range(plateau_start, first_drop):
        x = df.iat[i, rh_col]
        val = float(rise_fn(x)) if i <= peak_idx else float(drop_fn(x))
        df.iat[i, col] = round(val, 3)

    return {
        "rise_coeffs": rise_coeffs,
        "drop_popt": drop_popt,
        "drop_x0": drop_x0,
        "peak_idx": peak_idx,
        "peak_time": df.iat[peak_idx, df.columns.get_loc("measure_date")],
        "peak_value": df.iat[peak_idx, col],
    }


# ─── Main ────────────────────────────────────────────────────────────────────

df = pd.read_csv("data/co2_data.csv")
df["corrected_value"] = np.nan

# Anomaly 1
# Plateau 1 starts at CSV line 1065 → pandas iloc 1063
# First drop at CSV line 1469 → pandas iloc 1467
info1 = fill_anomaly(df, plateau_start=1063, first_drop=1467)

# Anomaly 2
# Plateau 2 starts at CSV line 1637 → pandas iloc 1635
# First drop at CSV line 1680 → pandas iloc 1678
info2 = fill_anomaly(df, plateau_start=1635, first_drop=1678)

# ─── Stacked reconstruction ──────────────────────────────────────────────────

def build_stacked(df, purge_events):
    """
    Create stacked_value: every rising segment is shifted so its starting
    minimum aligns with the previous segment's peak.

    Each purge_event is a dict with:
      peak_idx   : pandas iloc index of the estimated peak (from corrected_value)
      search_start : start of window to find the post-purge minimum
      search_end   : end of that window (exclusive)
    """
    # Base signal: use corrected_value where the sensor was saturated, else actual value
    cv_col = df.columns.get_loc("corrected_value")
    val_col = df.columns.get_loc("value")
    df["stacked_value"] = df["corrected_value"].combine_first(df["value"])

    sv_col = df.columns.get_loc("stacked_value")

    for event in purge_events:
        peak_idx = event["peak_idx"]
        s, e = event["search_start"], event["search_end"]

        # Peak of this segment (already includes any prior offsets)
        peak_val = float(df.iat[peak_idx, sv_col])

        # Minimum of the post-purge drop window (use raw value to locate position)
        window_raw = df.iloc[s:e]["value"]
        min_pos = int(window_raw.idxmin())   # label == iloc for default RangeIndex
        min_val = float(df.iat[min_pos, sv_col])  # includes prior offsets

        offset = peak_val - min_val

        # Shift everything from the minimum onwards
        df.iloc[min_pos:, sv_col] = df.iloc[min_pos:, sv_col] + offset

        print(
            f"  Purge: peak={peak_val:.1f} ppm  |  "
            f"drop min={min_val:.1f} ppm  |  "
            f"offset=+{offset:.1f} ppm  |  "
            f"new segment starts at row {min_pos} ({df.iat[min_pos, df.columns.get_loc('measure_date')]})"
        )

    return df


purge_events = [
    # Purge 1: peak at pandas 1465, search rows 1467–1499
    {"peak_idx": 1465, "search_start": 1467, "search_end": 1500},
    # Purge 2: peak at pandas 1676, search rows 1678–end
    {"peak_idx": 1676, "search_start": 1678, "search_end": len(df)},
]

print("Building stacked reconstruction...")
df = build_stacked(df, purge_events)

df.to_csv("data/co2_data.csv", index=False)

# ─── Report ──────────────────────────────────────────────────────────────────

print("Anomaly 1")
print(f"  Rising polynomial (degree 2):  {np.poly1d(info1['rise_coeffs'])}")
print(f"  Drop exp decay — a={info1['drop_popt'][0]:.2f}  b={info1['drop_popt'][1]:.4f}  c={info1['drop_popt'][2]:.2f}  x0={info1['drop_x0']:.3f}")
print(f"  Estimated peak: {info1['peak_value']:.1f} ppm  at  {info1['peak_time']}")

print()
print("Anomaly 2")
print(f"  Rising polynomial (degree 2):  {np.poly1d(info2['rise_coeffs'])}")
print(f"  Drop exp decay — a={info2['drop_popt'][0]:.2f}  b={info2['drop_popt'][1]:.4f}  c={info2['drop_popt'][2]:.2f}  x0={info2['drop_x0']:.3f}")
print(f"  Estimated peak: {info2['peak_value']:.1f} ppm  at  {info2['peak_time']}")

print()
print("Saved data/co2_data.csv")
