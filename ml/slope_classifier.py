import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect("data/greenhouse.duckdb")

df = con.execute("""
    SELECT
        id_evento,
        tiempo_relativo,
        humedad,
        humedad_externa,
        franja
    FROM vw_execution_v_metrics_bi
""").df()

def slope(x, y):
    if len(x) < 3:
        return np.nan
    return np.polyfit(x, y, 1)[0]

MIN_PRE = 10
MIN_POST = 20

features = []

for event_id, g in df.groupby("id_evento"):
    pre = g[(g.tiempo_relativo < 0)]
    post = g[(g.tiempo_relativo >= 0)]

    if len(pre) < MIN_PRE or len(post) < MIN_POST:
        continue  # drop low-quality events

    s_int_pre = slope(pre.tiempo_relativo, pre.humedad)
    s_ext_pre = slope(pre.tiempo_relativo, pre.humedad_externa)
    s_rel_pre = s_int_pre - s_ext_pre

    s_int_post = slope(post.tiempo_relativo, post.humedad)
    s_ext_post = slope(post.tiempo_relativo, post.humedad_externa)
    s_rel_post = s_int_post - s_ext_post

    delta_detach = s_rel_post - s_rel_pre

    features.append({
        "id_evento": event_id,
        "n_pre": len(pre),
        "n_post": len(post),
        "franja": g.franja.iloc[0],
        "slope_internal_pre": s_int_pre,
        "slope_external_pre": s_ext_pre,
        "slope_relative_pre": s_rel_pre,
        "slope_internal_post": s_int_post,
        "slope_external_post": s_ext_post,
        "slope_relative_post": s_rel_post,
        "delta_detach": delta_detach,
    })

features_df = pd.DataFrame(features)
EPS = 0.02  # %RH per minute (tune later)

def classify(row):

    if abs(row.delta_detach) < EPS:
        return "0"

    if row.slope_relative_post > 0:
        return "IF" if row.delta_detach > 0 else "IS"
    else:
        return "DF" if row.delta_detach < 0 else "DS"

features_df["class_v0"] = features_df.apply(classify, axis=1)

con.register("features_df", features_df)
con.execute("""
CREATE OR REPLACE TABLE event_humidity_features AS
SELECT *
FROM features_df
""")
