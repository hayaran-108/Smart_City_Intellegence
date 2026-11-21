#!/usr/bin/env python3
"""
Project 18 - Multi-Dataset Smart City Intelligence
==================================================
Libraries : numpy + pandas ONLY (no other imports at all).

QUESTION
    Which wards (and which hours of the day) carry the heaviest combined
    traffic / air-pollution / incident burden, does bus service offset it, and
    what would a targeted peak-hour traffic cut achieve?

HOW TO RUN
    python smart_city_intelligence.py
    Input CSVs are read from the current folder as data_<name>.csv (demo files are created if absent).
    Results are written as out_<name>.csv / out_recommendation.md. Change DATA_PREFIX / OUT_PREFIX below.

REAL DATA
    The pipeline ALWAYS reads CSV files named DATA_PREFIX + <file>. If they are missing it
    writes a synthetic demo set first (clearly labelled, NOT real). To use real
    open data, export your portal downloads to the same file names / columns
    (see SOURCES below; e.g. data_traffic_counts.csv), then re-run. Nothing else changes.
"""
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
SEED = 2025
DATA_PREFIX = "/content/ward_dataset_1500_rows.csv"                  # input files:  data_traffic_counts.csv, ...
OUT_PREFIX = "out_"                    # output files: out_07_composite_by_ward.csv, ...
START = "2025-03-01"
DAYS = 28
N_WARDS = 12
UTC_OFFSET = pd.Timedelta("5h30min")   # AQ feed is UTC; city is IST (UTC+05:30)
NO2_THRESHOLD = 90.0                   # ug/m3, illustrative hourly exceedance level (set to your standard)
PEAK_HOURS = [7, 8, 9, 10, 17, 18, 19, 20]
TRAFFIC_CUT = 0.15                     # scenario: 15% cut in peak-hour traffic
N_BOOT = 400
N_WEIGHT_DRAWS = 5000

# ----------------------------------------------------------------------------
# STEP 1 - source / licence register (EDIT for your real portals)
# ----------------------------------------------------------------------------
SOURCES = pd.DataFrame([
    dict(dataset="traffic", file="traffic_counts.csv", provider="City traffic-sensor open-data portal",
         licence="<<fill in: e.g. CC-BY 4.0 / ODbL / OGL>>", native_grain="sensor x hour",
         key="sensor_id -> ward_id (sensor_lookup.csv)", unit="vehicles/hour", tz="local"),
    dict(dataset="air_quality", file="air_quality.csv", provider="CPCB / city AQ monitoring open data",
         licence="<<fill in>>", native_grain="station x hour",
         key="station_id -> ward_id (station_lookup.csv)", unit="ug/m3 (some feeds mg/m3)", tz="UTC"),
    dict(dataset="weather", file="weather.csv", provider="National met service / reanalysis open data",
         licence="<<fill in>>", native_grain="city x hour", key="hour", unit="degF, m/s, mm", tz="local"),
    dict(dataset="transport", file="bus_boardings.csv", provider="Transit agency open data (GTFS-derived)",
         licence="<<fill in>>", native_grain="stop x hour",
         key="stop_id -> ward_ref (stops.csv, 'ward_3' style)", unit="boardings", tz="local"),
    dict(dataset="incidents", file="incidents.csv", provider="Police / emergency-response open data",
         licence="<<fill in>>", native_grain="event", key="ward_code", unit="count", tz="local"),
    dict(dataset="wards", file="wards.csv", provider="Census / municipal boundaries",
         licence="<<fill in>>", native_grain="ward", key="ward_id", unit="people, km2", tz="n/a"),
])
LOOKUPS = {"sensor_lookup": "sensor_lookup.csv", "station_lookup": "station_lookup.csv", "stops": "stops.csv"}


def h(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ----------------------------------------------------------------------------
# SYNTHETIC DEMO DATA (only used when real CSVs are absent)
# ----------------------------------------------------------------------------
def make_synthetic():
    rng = np.random.default_rng(SEED)
    ward_ids = [f"W{i:02d}" for i in range(1, N_WARDS + 1)]
    hours = pd.date_range(START, periods=24 * DAYS, freq="h")
    n = len(hours)
    hod, dow = hours.hour.to_numpy(), hours.dayofweek.to_numpy()

    wards = pd.DataFrame({
        "ward_id": ward_ids, "ward_name": [f"Ward {i}" for i in range(1, N_WARDS + 1)],
        "population": rng.integers(40_000, 160_000, N_WARDS),
        "area_km2": rng.uniform(4, 25, N_WARDS).round(1)})
    wards.to_csv(DATA_PREFIX + "wards.csv", index=False)

    # weather (degF on purpose: unit audit)
    temp_c = 24 + 6 * np.sin((hod - 9) / 24 * 2 * np.pi) + rng.normal(0, 1, n)
    wind = np.clip(2.5 + 1.2 * np.sin((hod - 14) / 24 * 2 * np.pi) + rng.normal(0, 0.8, n), 0.3, None)
    rain = np.where(rng.random(n) < 0.08, rng.exponential(2.0, n), 0.0)
    pd.DataFrame({"timestamp": hours.strftime("%Y-%m-%d %H:%M:%S"), "temp_f": (temp_c * 9 / 5 + 32).round(1),
                  "wind_ms": wind.round(2), "rain_mm": rain.round(2)}).to_csv(DATA_PREFIX + "weather.csv", index=False)

    prof = (0.3 + 0.7 * (np.exp(-((hod - 9) / 2) ** 2) + np.exp(-((hod - 18) / 2.5) ** 2))
            + 0.2 * ((hod >= 10) & (hod <= 17)))
    wk = np.where(dow >= 5, 0.8, 1.0)
    level = rng.uniform(300, 1200, N_WARDS)
    T = level[:, None] * prof[None, :] * wk[None, :] * np.where(rain > 0.5, 0.9, 1.0)[None, :]

    # traffic sensors: 3 per ward + 1 orphan sensor; duplicates; outage; random gaps
    rows, sl = [], []
    for w in range(N_WARDS):
        for k in range(3):
            sid = f"S{w * 3 + k + 1:02d}"
            sl.append((sid, ward_ids[w]))
            v = T[w] * rng.uniform(0.8, 1.2) * rng.lognormal(0, 0.1, n)
            rows.append(pd.DataFrame({"sensor_id": sid, "timestamp": hours.strftime("%Y-%m-%d %H:%M:%S"),
                                      "vehicles": v.round(0)}))
    rows.append(pd.DataFrame({"sensor_id": "S99", "timestamp": hours.strftime("%Y-%m-%d %H:%M:%S"),
                              "vehicles": rng.integers(100, 500, n)}))
    tr = pd.concat(rows, ignore_index=True)
    tr = tr[rng.random(len(tr)) > 0.03]
    tr = tr[~((tr.sensor_id == "S05") & (pd.to_datetime(tr.timestamp) >= pd.Timestamp(START) + pd.Timedelta(days=10))
              & (pd.to_datetime(tr.timestamp) < pd.Timestamp(START) + pd.Timedelta(days=17)))]
    tr = pd.concat([tr, tr.sample(25, random_state=1)], ignore_index=True)
    tr.to_csv(DATA_PREFIX + "traffic_counts.csv", index=False)
    pd.DataFrame(sl, columns=["sensor_id", "ward_id"]).to_csv(DATA_PREFIX + "sensor_lookup.csv", index=False)

    # air quality: stations only in 10 of 12 wards. Ward background is NEGATIVELY related to ward
    # traffic level on purpose (busy central wards are better ventilated) -> ecological fallacy demo.
    bg = 25 - 0.025 * (level - 750) + rng.normal(0, 3, N_WARDS)
    no2 = 15 + 0.035 * T + 20 / wind[None, :] + bg[:, None] - 1.5 * np.minimum(rain, 3)[None, :]
    pm25 = 18 + 0.02 * T + 30 / wind[None, :] + bg[:, None] - 2 * np.minimum(rain, 3)[None, :]
    covered = [w for w in range(N_WARDS) if ward_ids[w] not in ("W04", "W09")]
    aq, st = [], []
    for i, w in enumerate(covered):
        sid = f"AQ{i + 1:02d}"
        st.append((sid, ward_ids[w]))
        a = no2[w] + rng.normal(0, 3, n)
        b = pm25[w] + rng.normal(0, 4, n)
        unit = "ug/m3"
        if sid == "AQ07":                      # unit trap: mg/m3
            a, b, unit = a / 1000, b / 1000, "mg/m3"
        d = pd.DataFrame({"station_id": sid,
                          "timestamp_utc": (hours - UTC_OFFSET).strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "no2": a.round(5), "pm25": b.round(5), "unit": unit})
        d.loc[rng.random(n) < 0.02, ["no2", "pm25"]] = np.nan
        aq.append(d)
    pd.concat(aq, ignore_index=True).to_csv(DATA_PREFIX + "air_quality.csv", index=False)
    pd.DataFrame(st, columns=["station_id", "ward_id"]).to_csv(DATA_PREFIX + "station_lookup.csv", index=False)

    # bus boardings: 5 stops/ward, 'ward_3' style keys, 2 stops in non-existent ward_13
    tlevel = rng.uniform(20, 120, N_WARDS)
    stops, brd = [], []
    for w in range(N_WARDS):
        for k in range(5):
            sid = f"B{w * 5 + k + 1:03d}"
            stops.append((sid, f"ward_{w + 1}"))
            lam = tlevel[w] * prof * wk * np.where(rain > 0.5, 0.85, 1.0) / 3
            brd.append(pd.DataFrame({"stop_id": sid, "timestamp": hours.strftime("%Y-%m-%d %H:%M:%S"),
                                     "boardings": rng.poisson(lam)}))
    stops += [("B901", "ward_13"), ("B902", "ward_13")]
    brd.append(pd.DataFrame({"stop_id": ["B901"] * n, "timestamp": hours.strftime("%Y-%m-%d %H:%M:%S"),
                             "boardings": rng.poisson(10, n)}))
    pd.concat(brd, ignore_index=True).to_csv(DATA_PREFIX + "bus_boardings.csv", index=False)
    pd.DataFrame(stops, columns=["stop_id", "ward_ref"]).to_csv(DATA_PREFIX + "stops.csv", index=False)

    # incidents: Poisson counts expanded to events; some null / invalid ward codes
    lam = (wards.population.to_numpy()[:, None] / 100_000) * 0.35 * (T / T.mean()) * (1 + 0.6 * (rain > 0))[None, :]
    cnt = rng.poisson(lam)
    w_idx, h_idx = np.nonzero(cnt)
    rep = cnt[w_idx, h_idx]
    w_idx, h_idx = np.repeat(w_idx, rep), np.repeat(h_idx, rep)
    ts = hours[h_idx] + pd.to_timedelta(rng.integers(0, 3600, len(h_idx)), unit="s")
    inc = pd.DataFrame({"incident_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "ward_code": np.array(ward_ids)[w_idx],
                        "type": rng.choice(["collision", "breakdown", "road_hazard", "other"], len(h_idx)),
                        "severity": rng.choice([1, 2, 3], len(h_idx), p=[.6, .3, .1])})
    bad = rng.random(len(inc))
    inc.loc[bad < 0.01, "ward_code"] = np.nan
    inc.loc[(bad >= 0.01) & (bad < 0.015), "ward_code"] = "W99"
    inc.to_csv(DATA_PREFIX + "incidents.csv", index=False)
    print(f"[demo] synthetic CSVs written as {DATA_PREFIX}*.csv  (NOT real data)")


# ----------------------------------------------------------------------------
# HELPERS: audit + safe merging
# ----------------------------------------------------------------------------
def audit(name, df, time_col=None, key=None):
    rec = {"dataset": name, "rows": len(df), "cols": df.shape[1],
           "null_pct": float(round(100 * df.isna().mean().mean(), 2)),
           "dup_rows_on_key": int(df.duplicated(key).sum()) if key else int(df.duplicated().sum())}
    if time_col:
        t = pd.to_datetime(df[time_col], utc=False, errors="coerce")
        rec.update(t_min=str(t.min()), t_max=str(t.max()), unparsable_ts=int(t.isna().sum()))
    print(f"-- {name}: {rec['rows']:,} rows | dtypes: " + ", ".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items()))
    print("   ", {k: v for k, v in rec.items() if k not in ('dataset',)})
    return rec


def key_report(name, left, right):
    L, R = set(left.dropna().unique()), set(right.dropna().unique())
    rec = {"check": name, "left_keys": len(L), "right_keys": len(R), "matched": len(L & R),
           "left_only": sorted(L - R)[:10], "right_only": sorted(R - L)[:10]}
    print(f"   key check [{name}] matched={rec['matched']} left_only={rec['left_only']} right_only={rec['right_only']}")
    return rec


UNMATCHED = []


def merge_report(left, right, on, validate, name, how="left"):
    """Merge with cardinality validation and an explicit unmatched-row report."""
    m = left.merge(right, on=on, how=how, validate=validate, indicator=True)
    vc = {k: int(v) for k, v in m["_merge"].value_counts().items() if v}
    print(f"   merge [{name}] validate={validate} -> {vc}")
    un = m[m["_merge"] != "both"].drop(columns="_merge").copy()
    if len(un):
        un.insert(0, "merge_name", name)
        UNMATCHED.append(un.head(200))
    return m.drop(columns="_merge")


# ----------------------------------------------------------------------------
# NumPy helpers
# ----------------------------------------------------------------------------
def robust_z(x):
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    return (x - med) / np.where(mad == 0, 1, mad)


def scale_0_100(x, clip=3.0):
    return (np.clip(robust_z(x), -clip, clip) + clip) / (2 * clip) * 100


def ranks_desc(score):
    """rank 1 = highest score (worst burden); works on 1-D or (draws, items)."""
    return (-score).argsort(axis=-1).argsort(axis=-1) + 1


def spearman(a, b):
    return float(np.corrcoef(a, b)[0, 1])  # inputs are already ranks


def residualise(y, C):
    return y - C @ np.linalg.lstsq(C, y, rcond=None)[0]


def decompose_corr(df, x, y):
    """Pooled vs between-ward (ecological) vs within-ward correlation."""
    d = df[["ward_id", x, y]].dropna()
    means = d.groupby("ward_id")[[x, y]].mean()
    dm = d[[x, y]] - d.groupby("ward_id")[[x, y]].transform("mean")
    with np.errstate(all="ignore"):
        return {"pair": f"{x} ~ {y}", "n_wards": len(means), "pooled": d[x].corr(d[y]),
                "between_ward": means[x].corr(means[y]), "within_ward": dm[x].corr(dm[y])}


def fe_ols(df, y, xcols, hod_dummies):
    """Ward fixed-effects OLS via numpy lstsq + bootstrap over DAYS (cluster resampling)."""
    X = df[xcols].astype(float).copy()
    if hod_dummies:
        X = pd.concat([X, pd.get_dummies(df["hod"], prefix="h", drop_first=True, dtype=float)], axis=1)
    Xd = (X - X.groupby(df["ward_id"]).transform("mean")).to_numpy()
    yd = (df[y] - df.groupby("ward_id")[y].transform("mean")).to_numpy()
    beta = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    day = df["day"].to_numpy()
    days = np.unique(day)
    XtX = np.stack([Xd[day == d].T @ Xd[day == d] for d in days])
    Xty = np.stack([Xd[day == d].T @ yd[day == d] for d in days])
    rng = np.random.default_rng(SEED)
    boots = np.empty((N_BOOT, Xd.shape[1]))
    for b in range(N_BOOT):
        idx = rng.integers(0, len(days), len(days))
        boots[b] = np.linalg.lstsq(XtX[idx].sum(0), Xty[idx].sum(0), rcond=None)[0]
    return beta, boots


# ----------------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 140, "display.max_columns", 30)

    needed = list(SOURCES.file) + list(LOOKUPS.values())
    try:
        for f in needed:
            open(DATA_PREFIX + f).close()
    except FileNotFoundError:
        make_synthetic()

    # ---------------- 1. question + sources -----------------------------------
    h("1. QUESTION & SOURCE REGISTER")
    print("Question: which wards/hours carry the heaviest combined traffic-pollution-incident burden,\n"
          "does bus service offset it, and what would a targeted peak-hour traffic cut achieve?\n")
    print(SOURCES.to_string(index=False))
    SOURCES.to_csv(OUT_PREFIX + "01_source_register.csv", index=False)

    raw = {k: pd.read_csv(DATA_PREFIX + f) for k, f in zip(SOURCES.dataset, SOURCES.file)}
    raw.update({k: pd.read_csv(DATA_PREFIX + f) for k, f in LOOKUPS.items()})

    # ---------------- 2. common grain -----------------------------------------
    h("2. COMMON GEOGRAPHY & TIME GRAIN")
    print("Geography: ward_id ('W01'..), the finest level shared by all sources (census/boundary table).\n"
          "Time     : local hour (naive, IST). Panel = ward x hour (analysis), rolled up to ward (ranking).\n"
          "Rules    : sensors/stops -> ward via lookup; AQ UTC -> local (+05:30); events -> floor to hour;\n"
          "           missing event hours = 0 incidents; missing sensor/AQ hours stay NaN (never zero-filled).")

    # ---------------- 3. audit -------------------------------------------------
    h("3. AUDIT: schema, units, coverage, keys")
    audits = [
        audit("traffic", raw["traffic"], "timestamp", ["sensor_id", "timestamp"]),
        audit("air_quality", raw["air_quality"], "timestamp_utc", ["station_id", "timestamp_utc"]),
        audit("weather", raw["weather"], "timestamp", ["timestamp"]),
        audit("transport", raw["transport"], "timestamp", ["stop_id", "timestamp"]),
        audit("incidents", raw["incidents"], "incident_time"),
        audit("wards", raw["wards"], None, ["ward_id"]),
    ]
    pd.DataFrame(audits).to_csv(OUT_PREFIX + "03_schema_audit.csv", index=False)

    wards = raw["wards"].copy()
    ward_ids = wards["ward_id"].tolist()
    print("\nUnit audit:")
    print("   air_quality.unit counts ->", raw["air_quality"]["unit"].value_counts().to_dict())
    print("   weather temp_f range    ->", raw["weather"]["temp_f"].agg(["min", "max"]).to_dict(), "(degF -> convert to degC)")
    print("\nKey compatibility:")
    kr = [key_report("traffic.sensor_id vs sensor_lookup", raw["traffic"].sensor_id, raw["sensor_lookup"].sensor_id),
          key_report("air_quality.station_id vs station_lookup", raw["air_quality"].station_id, raw["station_lookup"].station_id),
          key_report("transport.stop_id vs stops", raw["transport"].stop_id, raw["stops"].stop_id),
          key_report("stops.ward_ref (raw) vs wards.ward_id", raw["stops"].ward_ref, wards.ward_id),
          key_report("incidents.ward_code vs wards", raw["incidents"].ward_code, wards.ward_id)]
    print("   -> stops.ward_ref uses 'ward_3' style: normalise to 'W03' before joining.")
    pd.DataFrame(kr).to_csv(OUT_PREFIX + "03_key_compatibility.csv", index=False)

    # ---------------- 4. cleaning + merges ------------------------------------
    h("4. PANDAS MERGES with cardinality validation & unmatched-row reports")
    hours = pd.date_range(raw["weather"].timestamp.pipe(pd.to_datetime).min().floor("h"), periods=24 * DAYS, freq="h")

    # traffic
    tr = raw["traffic"].assign(hour=lambda d: pd.to_datetime(d.timestamp).dt.floor("h"))
    n_dup = int(tr.duplicated(["sensor_id", "hour"]).sum())
    tr = tr.drop_duplicates(["sensor_id", "hour"])
    print(f"   traffic: dropped {n_dup} duplicate sensor-hours")
    tr = merge_report(tr, raw["sensor_lookup"], "sensor_id", "many_to_one", "traffic -> sensor_lookup")
    tr = tr.dropna(subset=["ward_id"])
    traffic_wh = (tr.groupby(["ward_id", "hour"], as_index=False)
                    .agg(vehicles=("vehicles", "mean"), n_sensors=("sensor_id", "nunique")))

    # air quality (unit harmonisation, UTC -> local)
    aq = raw["air_quality"].copy()
    to_ug = np.where(aq["unit"].str.lower().eq("mg/m3"), 1000.0, 1.0)
    aq[["no2", "pm25"]] = aq[["no2", "pm25"]].mul(to_ug, axis=0)
    aq["hour"] = (pd.to_datetime(aq.timestamp_utc, utc=True).dt.tz_localize(None) + UTC_OFFSET).dt.floor("h")
    print(f"   air_quality: {int((to_ug != 1).sum())} rows converted mg/m3 -> ug/m3; UTC -> local applied")
    aq = aq.drop_duplicates(["station_id", "hour"])
    aq = merge_report(aq, raw["station_lookup"], "station_id", "many_to_one", "air_quality -> station_lookup").dropna(subset=["ward_id"])
    aq_wh = aq.groupby(["ward_id", "hour"], as_index=False).agg(no2=("no2", "mean"), pm25=("pm25", "mean"))

    # transport (key normalisation)
    stops = raw["stops"].copy()
    num = stops.ward_ref.str.extract(r"(\d+)")[0].astype(int)
    stops["ward_id"] = num.map(lambda i: f"W{i:02d}")
    bad_stops = stops[~stops.ward_id.isin(ward_ids)]
    print(f"   stops with ward not in ward table: {bad_stops.stop_id.tolist()} (excluded, logged)")
    UNMATCHED.append(bad_stops.assign(merge_name="stops -> wards"))
    stops = stops[stops.ward_id.isin(ward_ids)][["stop_id", "ward_id"]]
    bd = raw["transport"].assign(hour=lambda d: pd.to_datetime(d.timestamp).dt.floor("h"))
    bd = merge_report(bd, stops, "stop_id", "many_to_one", "boardings -> stops").dropna(subset=["ward_id"])
    transit_wh = bd.groupby(["ward_id", "hour"], as_index=False).agg(boardings=("boardings", "sum"))

    # incidents
    inc = raw["incidents"].assign(hour=lambda d: pd.to_datetime(d.incident_time).dt.floor("h"))
    inc_chk = inc.merge(wards[["ward_id"]].assign(_ok=1), left_on="ward_code", right_on="ward_id", how="left", validate="many_to_one")
    n_bad = int(inc_chk["_ok"].isna().sum())
    print(f"   incidents: {n_bad} events with null/unknown ward_code excluded ({100 * n_bad / len(inc):.2f}%)")
    UNMATCHED.append(inc_chk[inc_chk["_ok"].isna()].drop(columns=["ward_id", "_ok"]).assign(merge_name="incidents -> wards").head(200))
    inc_wh = (inc_chk[inc_chk["_ok"].notna()].groupby(["ward_id", "hour"], as_index=False).size().rename(columns={"size": "incidents"}))

    # weather
    wx = raw["weather"].assign(hour=lambda d: pd.to_datetime(d.timestamp).dt.floor("h"))
    wx["temp_c"] = (wx.temp_f - 32) * 5 / 9
    wx = wx.drop_duplicates("hour")[["hour", "temp_c", "wind_ms", "rain_mm"]]

    # skeleton ward x hour; every merge is one_to_one (or many_to_one for weather)
    panel = pd.MultiIndex.from_product([ward_ids, hours], names=["ward_id", "hour"]).to_frame(index=False)
    for part, nm in [(traffic_wh, "traffic"), (aq_wh, "air_quality"), (transit_wh, "transport"), (inc_wh, "incidents")]:
        panel = merge_report(panel, part, ["ward_id", "hour"], "one_to_one", f"panel <- {nm}")
    panel = merge_report(panel, wx, "hour", "many_to_one", "panel <- weather")
    panel = merge_report(panel, wards, "ward_id", "many_to_one", "panel <- wards")
    panel["incidents"] = panel["incidents"].fillna(0)      # event data: absence = zero events
    panel["day"] = (panel.hour - hours[0]).dt.days
    panel["hod"] = panel.hour.dt.hour
    panel["weekend"] = (panel.hour.dt.dayofweek >= 5).astype(float)

    print("\nCoverage of the ward-hour panel (share of ward-hours with data):")
    cov = panel[["vehicles", "no2", "pm25", "boardings", "temp_c"]].notna().mean().round(3)
    print(cov.to_string())
    ward_cov = panel.groupby("ward_id")[["vehicles", "no2", "boardings"]].agg(lambda s: s.notna().mean()).round(2)
    print("\nWards with poor coverage:\n", ward_cov[(ward_cov < 0.8).any(axis=1)].to_string())

    # AQ gap handling: neighbour-ward imputation (ring adjacency demo; use real adjacency in practice), flagged
    no_aq = ward_cov.index[ward_cov["no2"] == 0].tolist()
    panel["aq_imputed"] = panel.ward_id.isin(no_aq)
    for col in ["no2", "pm25"]:
        wide = panel.pivot(index="ward_id", columns="hour", values=col).reindex(ward_ids)
        arr = wide.to_numpy().copy()
        for w in no_aq:
            i = ward_ids.index(w)
            nb = [arr[(i - 1) % len(ward_ids)], arr[(i + 1) % len(ward_ids)]]
            if all(np.isnan(arr[(i + s) % len(ward_ids)]).all() for s in (-1, 1)):
                continue
            arr[i] = np.nanmean(np.vstack(nb), axis=0)
        panel[col] = arr.ravel()                      # panel is ward-major, hour-minor
    print(f"\nAQ imputed (neighbour mean, flagged aq_imputed) for wards without a station: {no_aq}")

    pd.concat(UNMATCHED, ignore_index=True).to_csv(OUT_PREFIX + "04_unmatched_rows.csv", index=False)

    # ---------------- 5. NumPy metrics & scenario ------------------------------
    h("5. NUMPY: robust metrics (median/MAD), normalisation, scenario")
    obs = panel[~panel.aq_imputed].dropna(subset=["vehicles", "no2", "pm25", "wind_ms", "temp_c", "rain_mm"]).copy()
    v = obs.vehicles.to_numpy()
    print(f"   vehicles: mean={v.mean():.0f} median={np.median(v):.0f} MAD*1.4826={np.median(np.abs(v - np.median(v))) * 1.4826:.0f}")
    print(f"   NO2 p50/p95/p99 = {np.percentile(obs.no2, [50, 95, 99]).round(1)} ug/m3 (robust vs outliers)")
    obs["inv_wind"] = 1 / obs.wind_ms
    obs["veh100"] = obs.vehicles / 100

    # ---------------- 6. cross-domain patterns --------------------------------
    h("6. CROSS-DOMAIN ANALYSIS: confounding & ecological fallacy")
    print("a) Pooled vs between-ward vs within-ward correlations (different questions, different answers):")
    dec = pd.DataFrame([decompose_corr(obs, "vehicles", "no2"), decompose_corr(obs, "vehicles", "pm25"),
                        decompose_corr(panel, "boardings", "incidents"), decompose_corr(panel, "vehicles", "incidents"),
                        decompose_corr(panel, "rain_mm", "incidents")]).round(3)
    print(dec.to_string(index=False))
    print("   -> If between-ward and within-ward signs differ, ward averages must NOT be used to infer hourly (or individual) effects.")

    print("\nb) Confounding: shared diurnal cycle, weather and weekend drive both traffic and NO2.")
    C = pd.concat([pd.Series(1.0, index=obs.index, name="const"), obs[["wind_ms", "temp_c", "rain_mm", "weekend"]],
                   pd.get_dummies(obs.hod, prefix="h", drop_first=True, dtype=float),
                   pd.get_dummies(obs.ward_id, prefix="w", drop_first=True, dtype=float)], axis=1).to_numpy()
    r_raw = np.corrcoef(obs.vehicles, obs.no2)[0, 1]
    r_par = np.corrcoef(residualise(obs.vehicles.to_numpy(), C), residualise(obs.no2.to_numpy(), C))[0, 1]
    print(f"   corr(vehicles, NO2) raw = {r_raw:.3f} | partial (ward + weather + hour-of-day + weekend removed) = {r_par:.3f}")

    print("\nc) Ward fixed-effects regression of NO2 (coefficient per +100 veh/h/sensor), bootstrap by day:")
    xs = ["veh100", "inv_wind", "rain_mm", "temp_c"]
    res = {}
    for label, hd in [("A: no hour controls", False), ("B: + hour-of-day dummies", True)]:
        beta, boots = fe_ols(obs, "no2", xs, hd)
        lo, hi = np.percentile(boots[:, 0], [2.5, 97.5])
        res[label] = (beta[0], lo, hi)
        print(f"   spec {label:26s} beta_traffic = {beta[0]:6.3f}  95% CI [{lo:6.3f}, {hi:6.3f}]")
    beta_B, boots_B = fe_ols(obs, "no2", xs, True)
    print("   -> Association only. Causal reading needs a design (e.g. natural experiment); here controls are limited.")

    sc = panel.dropna(subset=["vehicles", "no2"]).copy()
    base_no2 = sc.no2.to_numpy()
    sc_peak = sc.hod.isin(PEAK_HOURS).to_numpy()

    def run_scenario(target):
        t_rows = sc.ward_id.isin(target).to_numpy()
        d_veh = np.where(t_rows & sc_peak, -TRAFFIC_CUT * sc.vehicles.to_numpy(), 0.0)
        d_no2 = beta_B[0] * d_veh / 100
        boot_mean = (boots_B[:, 0][:, None] * d_veh[None, t_rows] / 100).mean(1)   # (B,) NumPy broadcasting
        before, after = (base_no2[t_rows] > NO2_THRESHOLD).mean(), ((base_no2 + d_no2)[t_rows] > NO2_THRESHOLD).mean()
        return dict(target_wards=",".join(target), traffic_cut=TRAFFIC_CUT, beta_B=beta_B[0],
                    mean_no2_change=d_no2[t_rows].mean(), ci_lo=np.percentile(boot_mean, 2.5),
                    ci_hi=np.percentile(boot_mean, 97.5), peak_hour_change=d_no2[t_rows & sc_peak].mean(),
                    exceed_before=before, exceed_after=after)

    def show(r):
        print(f"   targets [{r['target_wards']}]: mean NO2 change {r['mean_no2_change']:.2f} ug/m3 "
              f"(95% CI {r['ci_lo']:.2f} to {r['ci_hi']:.2f}); peak-hour change {r['peak_hour_change']:.2f}; "
              f"hours > {NO2_THRESHOLD:.0f}: {r['exceed_before']:.1%} -> {r['exceed_after']:.1%}")

    print("\nd) Scenario: cut peak-hour traffic by %d%% (hours %s)" % (int(TRAFFIC_CUT * 100), PEAK_HOURS))
    busiest = panel.groupby("ward_id").vehicles.mean().sort_values(ascending=False).index[:3].tolist()
    scen_busy = run_scenario(busiest)
    print("   Option 1 - busiest wards by traffic:")
    show(scen_busy)
    dec.to_csv(OUT_PREFIX + "06_correlation_decomposition.csv", index=False)

    # ---------------- 7. composite indicator ----------------------------------
    h("7. COMPOSITE INDICATOR & ALTERNATIVE WEIGHTS")
    wk_factor = len(hours) / 168
    ward = panel.groupby("ward_id").agg(
        traffic=("vehicles", "mean"), no2=("no2", "mean"), pm25_p95=("pm25", lambda s: np.nanpercentile(s, 95)),
        incidents=("incidents", "sum"), boardings=("boardings", "sum"), population=("population", "first"),
        aq_imputed=("aq_imputed", "first"))
    ward["incident_rate"] = ward.incidents / ward.population * 1e4 / wk_factor       # per 10k pop per week
    ward["boardings_per_1000"] = ward.boardings / ward.population * 1e3 / wk_factor
    ind = ["traffic", "no2", "pm25_p95", "incident_rate", "boardings_per_1000"]
    S = scale_0_100(ward[ind].to_numpy(float))
    S[:, ind.index("boardings_per_1000")] = 100 - S[:, ind.index("boardings_per_1000")]   # protective -> invert
    print("Indicators (robust-z, clipped +/-3, scaled 0-100; higher = worse; transit inverted).")

    schemes = {
        "baseline_equal":     [.2, .2, .2, .2, .2],
        "health_focus":       [.10, .35, .35, .10, .10],
        "safety_focus":       [.15, .10, .10, .55, .10],
        "congestion_focus":   [.45, .15, .10, .15, .15],
        "no_transit_credit":  [.25, .25, .25, .25, 0.0],
    }
    scores = {k: S @ np.array(w) for k, w in schemes.items()}
    base_rank = ranks_desc(scores["baseline_equal"])
    out = ward[["traffic", "no2", "pm25_p95", "incident_rate", "boardings_per_1000", "aq_imputed"]].round(2).copy()
    for k in schemes:
        out[f"rank_{k}"] = ranks_desc(scores[k])
    out["score_baseline"] = scores["baseline_equal"].round(1)
    print(out.sort_values("rank_baseline_equal").to_string())

    print("\nSpearman vs baseline (named schemes):")
    for k in schemes:
        print(f"   {k:18s} {spearman(base_rank, ranks_desc(scores[k])):.3f}")
    # scaling sensitivity: min-max instead of robust z
    mm = (ward[ind] - ward[ind].min()) / (ward[ind].max() - ward[ind].min()) * 100
    mm["boardings_per_1000"] = 100 - mm["boardings_per_1000"]
    print(f"   {'minmax_scaling':18s} {spearman(base_rank, ranks_desc(mm.to_numpy() @ np.full(5, .2))):.3f}")
    # leave-one-indicator-out
    print("Leave-one-indicator-out (Spearman vs baseline):")
    for j, nme in enumerate(ind):
        w = np.full(5, .25); w[j] = 0
        print(f"   drop {nme:20s} {spearman(base_rank, ranks_desc(S @ w)):.3f}")

    rng = np.random.default_rng(SEED)
    W = rng.dirichlet(np.ones(len(ind)), size=N_WEIGHT_DRAWS)              # random weights on the simplex
    R = ranks_desc(W @ S.T)
    stab = pd.DataFrame({"mean_rank": R.mean(0), "rank_p05": np.percentile(R, 5, axis=0),
                         "rank_p95": np.percentile(R, 95, axis=0), "P_top3": (R <= 3).mean(0)}, index=ward.index).round(2)
    print(f"\nMonte-Carlo weights ({N_WEIGHT_DRAWS} Dirichlet draws): rank stability")
    print(stab.sort_values("mean_rank").to_string())

    out.to_csv(OUT_PREFIX + "07_composite_by_ward.csv")
    stab.to_csv(OUT_PREFIX + "07_rank_stability.csv")

    # hour-of-day burden profile (robust-z of city mean by hour)
    hp = panel.groupby("hod").agg(traffic=("vehicles", "mean"), no2=("no2", "mean"), incidents=("incidents", "sum"))
    hp["burden_z"] = robust_z(hp[["traffic", "no2", "incidents"]].to_numpy()).mean(1)
    hp.round(2).to_csv(OUT_PREFIX + "07_hour_profile.csv")
    worst_hours = hp.burden_z.sort_values(ascending=False).index[:4].tolist()

    # ---------------- 8. recommendation + reproducibility ---------------------
    h("8. INTEGRATED RECOMMENDATION")
    robust_top = stab.index[(stab.P_top3 >= 0.5)].tolist()
    top_base = out.sort_values("rank_baseline_equal").index[:3].tolist()
    print("Scenario Option 2 - composite-priority wards (traffic cut applied where burden is highest):")
    scen_prio = run_scenario(robust_top or top_base)
    show(scen_prio)
    pd.DataFrame([scen_busy, scen_prio]).to_csv(OUT_PREFIX + "06_scenarios.csv", index=False)
    ecological = dec.iloc[0]
    rec = [
        f"Priority wards (baseline top-3): {', '.join(top_base)}. Rank-robust under random weights (P(top-3) >= 50%): {', '.join(robust_top) or 'none'}.",
        f"Worst hours of day (city burden index): {sorted(worst_hours)}. Focus interventions there, not all day.",
        f"Peak-hour traffic -{int(TRAFFIC_CUT * 100)}% in {scen_prio['target_wards']} lowers modelled NO2 by "
        f"{abs(scen_prio['mean_no2_change']):.2f} ug/m3 on average (95% CI {abs(scen_prio['ci_hi']):.2f}-{abs(scen_prio['ci_lo']):.2f}) and cuts "
        f"threshold-exceedance hours {scen_prio['exceed_before']:.1%} -> {scen_prio['exceed_after']:.1%} (association-based estimate).",
        f"Caution: traffic-NO2 correlation is pooled {ecological['pooled']:.2f}, within-ward {ecological['within_ward']:.2f}, "
        f"between-ward {ecological['between_ward']:.2f}. Use within-ward evidence for hourly policy; do not read ward averages as hourly effects.",
        f"Data caveats: AQ imputed for {no_aq} (no station) - install/borrow monitors there before acting on those wards.",
    ]
    for r in rec:
        print(" *", r)
    open(OUT_PREFIX + "08_recommendation.md", "w").write("# Integrated recommendation\n\n" + "\n".join(f"- {r}" for r in rec) + "\n")

    h("REPRODUCIBILITY MANIFEST")
    fp = lambda df: int(pd.util.hash_pandas_object(df, index=False).sum() % (2 ** 63))   # content fingerprint
    rows = [("seed", SEED), ("numpy", np.__version__), ("pandas", pd.__version__), ("panel_rows", len(panel)),
            ("config", str(dict(START=START, DAYS=DAYS, NO2_THRESHOLD=NO2_THRESHOLD, PEAK_HOURS=PEAK_HOURS,
                                TRAFFIC_CUT=TRAFFIC_CUT, N_BOOT=N_BOOT, N_WEIGHT_DRAWS=N_WEIGHT_DRAWS,
                                UTC_OFFSET=str(UTC_OFFSET)))),
            ("steps", "read CSV -> audit -> clean/units/tz -> validated merges -> ward-hour panel -> metrics -> models -> composite -> report")]
    rows += [(f"input_fingerprint:{k}", fp(v)) for k, v in raw.items()]
    manifest = pd.DataFrame(rows, columns=["item", "value"])
    manifest.to_csv(OUT_PREFIX + "manifest.csv", index=False)
    print(manifest[~manifest.item.isin(["config", "steps"])].to_string(index=False))
    print(f"\nAll outputs written as {OUT_PREFIX}*")


if __name__ == "__main__":
    main()
