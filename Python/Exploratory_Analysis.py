"""Exploratory analysis and EV corridor reliability calculations."""

import os
# Limit OpenMP threads to prevent the Windows MKL K-Means warning.
os.environ["OMP_NUM_THREADS"] = "1"
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.ops import linemerge, substring, unary_union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from Data_Loading import (
    TABLE_DIR, OUTPUT_DIR, load_cleaned_datasets, ensure_output_directories
)


warnings.filterwarnings("ignore", category=FutureWarning)


METRIC_CRS = "EPSG:7755"
FAST_KW = 25
CORRIDOR_KM = 5
SEG_KM = 25
GAP_KM = 50
MIN_COMPONENT_KM = 5
MIN_DEMAND = 5000
USE_DATASET_FLAG = False
TARGET_HIGHWAYS = ["NH-44", "NH-48", "NH-16", "NH-66", "NH-544", "NH-52"]
YEARS = [str(year) for year in range(2017, 2027)]
DNHDD = "dadra and nagar haveli and daman and diu"

ALIASES = {
    "orissa": "odisha",
    "uttaranchal": "uttarakhand",
    "pondicherry": "puducherry",
    "nct of delhi": "delhi",
    "delhi nct": "delhi",
    "ut of dnh and dd": DNHDD,
    "dnh and dd": DNHDD,
    "dadra and nagar haveli": DNHDD,
    "daman and diu": DNHDD,
}


def make_key(series):
    return (
        series.astype("string").str.lower()
        .str.replace("&", "and", regex=False)
        .str.replace(r"[^a-z\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.replace("islands", "island", regex=False)
        .replace(ALIASES)
    )


def save_table(frame, filename, index=False):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / filename
    frame.to_csv(path, index=index)
    return path


def save_geojson(frame, filename):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / filename
    export_frame = frame.drop(
        columns=["date_created", "date_last_verified"], errors="ignore"
    )
    export_frame.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
    return path


def bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    return series.astype("string").str.strip().str.lower().isin(
        ["true", "1", "yes", "y", "operational"]
    )


def minmax(series):
    series = series.astype(float)
    low, high = series.min(), series.max()
    return (series - low) / (high - low) if high > low else series * 0


def longest_gap_km(group):
    best = 0.0
    for _, component in group.sort_values("start_km").groupby("component_id"):
        current = 0.0
        for is_gap, length in zip(component["is_gap"], component["length_km"]):
            current = current + length if is_gap else 0.0
            best = max(best, current)
    return best


def gap_distance(segments, fast_points):
    """Distance from each segment midpoint to the nearest fast charger on that highway."""
    distances = pd.Series(np.inf, index=segments.index, dtype=float)
    for road_name, indices in segments.groupby("road_name").groups.items():
        points = fast_points.loc[fast_points["road_name"] == road_name]
        if points.empty:
            continue
        tree = cKDTree(np.c_[points.geometry.x, points.geometry.y])
        nearest, _ = tree.query(segments.loc[indices, ["mid_x", "mid_y"]].values)
        distances.loc[indices] = nearest / 1000
    return distances


def run_eda():
    ensure_output_directories()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_cleaned_datasets()
    states_gdf = data["states"].copy()
    highways_gdf = data["highways"].copy()
    ev_registration = data["ev_registration"].copy()
    chargers_df = data["chargers"].copy()

    print("=" * 72)
    print("DATASET OVERVIEW")
    print("=" * 72)
    for name, frame in data.items():
        print(f"{name:18s} shape={frame.shape}")
        save_table(pd.DataFrame({
            "column": frame.columns.astype(str),
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "missing_count": [int(frame[col].isna().sum()) for col in frame.columns],
            "unique_count": [int(frame[col].nunique(dropna=True)) for col in frame.columns],
        }), f"{name}_column_profile.csv")

    quality_rows = []
    for name, frame in data.items():
        for column in frame.columns:
            quality_rows.append({
                "dataset": name,
                "column": str(column),
                "dtype": str(frame[column].dtype),
                "rows": len(frame),
                "missing_count": int(frame[column].isna().sum()),
                "missing_pct": round(float(frame[column].isna().mean() * 100), 2),
                "unique_count": int(frame[column].nunique(dropna=True)),
            })
    save_table(pd.DataFrame(quality_rows), "data_quality_report.csv")
    save_table(pd.DataFrame([
        {
            "dataset": name,
            "rows": len(frame),
            "columns": len(frame.columns),
            "duplicate_full_rows": int(frame.drop(columns="geometry", errors="ignore").duplicated().sum()),
        }
        for name, frame in data.items()
    ]), "duplicate_row_summary.csv")

    numeric_rows = []
    for name, frame in data.items():
        numeric = frame.select_dtypes(include=np.number)
        if not numeric.empty:
            summary = numeric.describe().T.reset_index().rename(columns={"index": "column"})
            summary.insert(0, "dataset", name)
            numeric_rows.append(summary)
    if numeric_rows:
        save_table(pd.concat(numeric_rows, ignore_index=True), "numeric_descriptive_statistics.csv")

    # EV registration trends and state-name matching.
    ev_registration.columns = ev_registration.columns.astype(str)
    missing_years = [year for year in YEARS if year not in ev_registration.columns]
    if missing_years:
        raise ValueError(f"EV registration data is missing required year columns: {missing_years}")
    if "State" not in ev_registration.columns or "Total" not in ev_registration.columns:
        raise ValueError("EV registration data must contain 'State' and 'Total' columns.")
    for year in YEARS + ["Total"]:
        ev_registration[year] = pd.to_numeric(ev_registration[year], errors="coerce")
    ev_registration["state_key"] = make_key(ev_registration["State"])
    states_gdf["state_key"] = make_key(states_gdf["state"])
    key2name = states_gdf.drop_duplicates("state_key").set_index("state_key")["state"].to_dict()

    yearly = ev_registration[YEARS].sum().rename_axis("year").reset_index(name="registrations")
    yearly["year"] = yearly["year"].astype(str)
    save_table(yearly, "ev_registrations_by_year.csv")
    save_table(
        ev_registration[["State"] + YEARS + ["Total"]].sort_values("Total", ascending=False),
        "ev_registration_by_state.csv",
    )
    mismatch = (ev_registration[YEARS].sum(axis=1) - ev_registration["Total"]).abs().sum()
    print("EV yearly-total mismatch (absolute sum):", round(float(mismatch), 2))
    print("States in boundaries but not EV registration:", sorted(set(states_gdf.state_key) - set(ev_registration.state_key)))
    print("States in EV registration but not boundaries:", sorted(set(ev_registration.state_key) - set(states_gdf.state_key)))

    # Prepare operational charger points and standardize fast-charging flags.
    if chargers_df.empty:
        raise ValueError("The cleaned charger dataset is empty.")
    for col in ["latitude", "longitude", "max_power_kw", "total_points"]:
        if col in chargers_df:
            chargers_df[col] = pd.to_numeric(chargers_df[col], errors="coerce")
    chargers_df["is_operational"] = bool_series(chargers_df["is_operational"])
    chargers_df["has_fast_charging"] = bool_series(chargers_df["has_fast_charging"])
    chargers_op = chargers_df.loc[chargers_df["is_operational"]].copy()
    chargers_op["max_power_kw"] = chargers_op["max_power_kw"].replace(0, np.nan)
    if USE_DATASET_FLAG:
        chargers_op["is_fast"] = chargers_op["has_fast_charging"]
    else:
        chargers_op["is_fast"] = chargers_op["max_power_kw"] >= FAST_KW

    chargers_gdf = gpd.GeoDataFrame(
        chargers_op,
        geometry=gpd.points_from_xy(chargers_op["longitude"], chargers_op["latitude"]),
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)

    states_m = states_gdf.to_crs(METRIC_CRS)
    st_small = states_m[["state_key", "geometry"]].rename(columns={"state_key": "state_geo"})
    chargers_join = gpd.sjoin(chargers_gdf, st_small, how="left", predicate="within")
    chargers_join = chargers_join.loc[~chargers_join.index.duplicated()].drop(columns="index_right")
    missing_state = chargers_join["state_geo"].isna()
    if missing_state.any():
        fix = gpd.sjoin_nearest(
            chargers_join.loc[missing_state].drop(columns=["state_geo"], errors="ignore"),
            st_small, how="left", max_distance=25_000
        )
        fix = fix.loc[~fix.index.duplicated()]
        chargers_join.loc[missing_state, "state_geo"] = fix["state_geo"]
    chargers_gdf = chargers_join.dropna(subset=["state_geo"]).copy()
    chargers_gdf["state_name"] = chargers_gdf["state_geo"].map(key2name)

    print("Operational chargers mapped to states:", len(chargers_gdf))
    print(f"Fast chargers used in analysis (>= {FAST_KW} kW):", int(chargers_gdf["is_fast"].sum()))
    charger_state_counts = chargers_gdf.groupby("state_name").size().rename("n_chargers").reset_index()
    save_table(charger_state_counts.sort_values("n_chargers", ascending=False), "chargers_by_state.csv")
    save_table(
        chargers_gdf.drop(columns="geometry").groupby("state_name").agg(
            n_chargers=("ID", "count"),
            n_fast=("is_fast", "sum"),
            median_power_kw=("max_power_kw", "median"),
            total_points=("total_points", "sum"),
        ).reset_index(),
        "charger_summary_by_state.csv",
    )
    save_geojson(chargers_gdf.drop(columns=["date_created", "date_last_verified"], errors="ignore"), "chargers_clean.geojson")

    # Combine EV demand, charger supply and state area.
    ev = ev_registration[["state_key", "2021", "2025", "Total"]].rename(columns={"Total": "ev_total"})
    ev["ev_cagr_21_25"] = np.where(
        ev["2021"] > 0, (ev["2025"] / ev["2021"]) ** (1 / 4) - 1, np.nan
    )
    ev = ev[["state_key", "ev_total", "ev_cagr_21_25"]]

    chg = (
        chargers_gdf.groupby("state_geo")
        .agg(
            n_chargers=("ID", "count"),
            n_fast=("is_fast", "sum"),
            total_points=("total_points", "sum"),
            median_kw=("max_power_kw", "median"),
        )
        .reset_index()
        .rename(columns={"state_geo": "state_key"})
    )
    master = (
        states_gdf[["state", "state_key", "area_km2", "geometry"]]
        .merge(ev, on="state_key", how="left")
        .merge(chg, on="state_key", how="left")
    )
    master[["n_chargers", "n_fast", "total_points"]] = master[
        ["n_chargers", "n_fast", "total_points"]
    ].fillna(0)
    master["ev_total"] = master["ev_total"].fillna(0)
    master["ev_demand"] = master["ev_total"]
    master["chargers_per_1000km2"] = master["n_chargers"] / master["area_km2"] * 1000
    master["evs_per_charger"] = master["ev_total"] / master["n_chargers"].clip(lower=1)
    master["evs_per_fast_charger"] = master["ev_demand"] / (master["n_fast"] + 1)
    master["fast_per_100k_ev"] = (
        master["n_fast"] / master["ev_demand"] * 1e5
    ).replace([np.inf, -np.inf], np.nan)
    med_ev = master["ev_demand"].median()
    med_fc = master["n_fast"].median()
    master["ev_desert"] = (master["ev_demand"] >= med_ev) & (master["n_fast"] < med_fc)
    save_table(
        master.drop(columns="geometry").sort_values("evs_per_fast_charger", ascending=False),
        "state_demand_supply_master.csv",
    )

    # Measure charger proximity to the national highway network.
    hw_all = highways_gdf.copy()
    if hw_all.crs is None:
        raise ValueError("Cleaned highway dataset has no CRS.")
    hw_all = hw_all.to_crs(METRIC_CRS)
    hw_all = hw_all.loc[
        hw_all.geometry.notna() & ~hw_all.geometry.is_empty,
        ["road_name", "road_type", "geometry"]
    ].explode(index_parts=False).reset_index(drop=True)
    hw_all = hw_all.loc[hw_all.geom_type == "LineString"].reset_index(drop=True)
    hw_all = hw_all.loc[~hw_all.geometry.to_wkb().duplicated()].reset_index(drop=True)
    if hw_all.empty:
        raise ValueError("No valid highway LineString geometries remain after preparation.")

    near = gpd.sjoin_nearest(
        chargers_gdf[["geometry"]], hw_all[["geometry"]],
        how="left", distance_col="distance_m"
    )
    near = near.loc[~near.index.duplicated()]
    chargers_gdf["dist_to_hwy_km"] = near["distance_m"] / 1000
    chargers_gdf["on_corridor"] = chargers_gdf["dist_to_hwy_km"] <= CORRIDOR_KM
    print(f"Share of mapped operational chargers within {CORRIDOR_KM} km of a highway:",
          round(float(chargers_gdf["on_corridor"].mean()), 3))

    # National highway segmentation and gap detection.
    segments_list = []
    for geometry in hw_all.geometry:
        length = geometry.length
        count = max(1, int(round(length / (SEG_KM * 1000))))
        step = length / count
        segments_list.extend(
            substring(geometry, index * step, (index + 1) * step)
            for index in range(count)
        )
    seg_nat = gpd.GeoDataFrame(geometry=segments_list, crs=METRIC_CRS)
    seg_nat["seg_km"] = seg_nat.length / 1000
    seg_nat = seg_nat.loc[seg_nat["seg_km"] > 0.5].reset_index(drop=True)
    midpoints = seg_nat.geometry.interpolate(0.5, normalized=True)
    mid_gdf = gpd.GeoDataFrame(geometry=midpoints, crs=METRIC_CRS)
    assigned = gpd.sjoin_nearest(mid_gdf, st_small, how="left")
    assigned = assigned.loc[~assigned.index.duplicated()]
    seg_nat["state_key"] = assigned["state_geo"]

    fast_chargers = chargers_gdf.loc[chargers_gdf["is_fast"]]
    if fast_chargers.empty:
        seg_nat["dist_fast_km"] = np.inf
    else:
        tree = cKDTree(np.c_[fast_chargers.geometry.x, fast_chargers.geometry.y])
        dist, _ = tree.query(np.c_[midpoints.x, midpoints.y])
        seg_nat["dist_fast_km"] = dist / 1000
    seg_nat["is_gap"] = seg_nat["dist_fast_km"] > GAP_KM
    seg_nat["gap_km"] = seg_nat["seg_km"] * seg_nat["is_gap"]
    national_gap_km = float(seg_nat["gap_km"].sum())
    print(
        f"National segments: {len(seg_nat):,}; gap segments: {int(seg_nat['is_gap'].sum()):,}; "
        f"uncovered distance: {national_gap_km:,.0f} km "
        f"({national_gap_km / seg_nat['seg_km'].sum():.1%})"
    )
    save_geojson(seg_nat, "national_segments_gap.geojson")

    # Prepare continuous components for the six target highways.
    hw6 = hw_all.loc[hw_all["road_name"].isin(TARGET_HIGHWAYS)].copy()
    found_roads = set(hw6["road_name"].dropna().unique())
    missing_roads = sorted(set(TARGET_HIGHWAYS) - found_roads)
    if missing_roads:
        print("Target highways not present in cleaned data:", missing_roads)
    if hw6.empty:
        raise ValueError(
            "None of the TARGET_HIGHWAYS were found in highways_clean.geojson. "
            "Check the road_name values and TARGET_HIGHWAYS in eda.py."
        )

    raw_km = highways_gdf.loc[
        highways_gdf["road_name"].isin(TARGET_HIGHWAYS)
    ].groupby("road_name")["length_km"].sum()

    def extract_lines(geometry):
        if geometry is None or geometry.is_empty:
            return []
        if geometry.geom_type == "LineString":
            return [geometry]
        if geometry.geom_type in ["MultiLineString", "GeometryCollection"]:
            lines = []
            for part in geometry.geoms:
                lines.extend(extract_lines(part))
            return lines
        return []

    component_rows = []
    for road, group in hw6.groupby("road_name"):
        union_geom = unary_union(group.geometry.values)
        if union_geom.geom_type == "MultiLineString":
            merged = linemerge(union_geom)
        else:
            merged = union_geom
        parts = extract_lines(merged)
        for component_id, line in enumerate(parts, start=1):
            if line.length > 0:
                component_rows.append({
                    "road_name": road, "component_id": component_id,
                    "length_km": line.length / 1000, "geometry": line
                })
    components = gpd.GeoDataFrame(component_rows, geometry="geometry", crs=METRIC_CRS)
    components["keep"] = components["length_km"] >= MIN_COMPONENT_KM
    road_parts = components.loc[components["keep"]].copy()
    if road_parts.empty:
        raise ValueError("No target-highway components meet MIN_COMPONENT_KM.")

    length_check = pd.DataFrame({
        "raw_summed_km": raw_km,
        "unique_km": components.groupby("road_name")["length_km"].sum(),
        "kept_km": road_parts.groupby("road_name")["length_km"].sum(),
        "pieces_kept": road_parts.groupby("road_name").size(),
    }).round(1).reset_index(names="road_name")
    save_table(length_check, "target_highway_length_check.csv")

    buffers = road_parts.dissolve(by="road_name")
    buffers["geometry"] = buffers.geometry.buffer(CORRIDOR_KM * 1000)
    buffers = buffers.reset_index()[["road_name", "geometry"]]
    corridor_chargers = gpd.sjoin(
        chargers_gdf, buffers, predicate="within", how="inner"
    ).drop(columns="index_right")
    fast_corridor_chargers = corridor_chargers.loc[corridor_chargers["is_fast"]].copy()

    if corridor_chargers.empty:
        cc_stats = pd.DataFrame(columns=["road_name", "all_chargers", "fast_chargers", "avg_power_kw"])
    else:
        cc_stats = corridor_chargers.groupby("road_name").agg(
            all_chargers=("ID", "nunique"),
            fast_chargers=("is_fast", "sum"),
            avg_power_kw=("max_power_kw", "mean"),
        ).round(1).reset_index()
    save_table(cc_stats, "corridor_charger_summary.csv")
    save_geojson(corridor_chargers, "corridor_chargers.geojson")
    save_geojson(fast_corridor_chargers, "fast_corridor_chargers.geojson")

    # Segment the target highways and measure same-highway fast-charger distances.
    seg_rows = []
    for _, row in road_parts.iterrows():
        length = row.geometry.length
        count = max(1, int(round(length / (SEG_KM * 1000))))
        step = length / count
        for index in range(count):
            start = index * step
            end = min((index + 1) * step, length)
            segment = substring(row.geometry, start, end)
            if segment.is_empty or segment.length <= 0:
                continue
            midpoint = segment.interpolate(0.5, normalized=True)
            seg_rows.append({
                "road_name": row["road_name"],
                "component_id": row["component_id"],
                "segment_id": index + 1,
                "start_km": start / 1000,
                "end_km": end / 1000,
                "length_km": segment.length / 1000,
                "mid_x": midpoint.x,
                "mid_y": midpoint.y,
                "geometry": segment,
            })
    segments = gpd.GeoDataFrame(seg_rows, geometry="geometry", crs=METRIC_CRS)
    if segments.empty:
        raise ValueError("No target-highway segments were created.")

    segments["nearest_fast_km"] = gap_distance(segments, fast_corridor_chargers)
    segments["is_gap"] = segments["nearest_fast_km"] > GAP_KM
    segments["gap_km"] = segments["length_km"] * segments["is_gap"]
    mids6 = gpd.GeoDataFrame(
        segments[["road_name"]],
        geometry=gpd.points_from_xy(segments["mid_x"], segments["mid_y"]),
        crs=METRIC_CRS,
    )
    j6 = gpd.sjoin_nearest(mids6, st_small, how="left")
    j6 = j6.loc[~j6.index.duplicated()]
    segments["state_key"] = j6["state_geo"]
    segments["state"] = segments["state_key"].map(key2name)
    print("Target highway segments:", len(segments))
    print("Gap segments:", int(segments["is_gap"].sum()))
    save_geojson(segments.drop(columns=["mid_x", "mid_y"]), "segments_6_highways_gap.geojson")

    # Highway scorecard.
    score_rows = []
    for road, group in segments.groupby("road_name"):
        score_rows.append({
            "road_name": road,
            "road_km": group["length_km"].sum(),
            "segments": len(group),
            "gap_segments": int(group["is_gap"].sum()),
            "gap_km": group["gap_km"].sum(),
            "longest_gap_km": longest_gap_km(group),
        })
    sc = pd.DataFrame(score_rows)
    sc["covered_pct"] = 100 * (1 - sc["gap_km"] / sc["road_km"])
    sc = sc.merge(cc_stats, on="road_name", how="left")
    sc[["all_chargers", "fast_chargers"]] = sc[["all_chargers", "fast_chargers"]].fillna(0)
    sc["fast_per_100km"] = sc["fast_chargers"] / sc["road_km"] * 100
    sc["rating"] = pd.cut(
        sc["covered_pct"], [-1, 70, 90, 101], labels=["Poor", "Patchy", "Reliable"]
    ).astype("string")
    sc = sc.sort_values("covered_pct", ascending=False).reset_index(drop=True).round(1)
    save_table(sc, "highway_scorecard.csv")

    # State-level gap priority for the six target highways.
    state_gap6 = segments.groupby("state").agg(
        highway_km=("length_km", "sum"),
        gap_segments=("is_gap", "sum"),
        gap_km=("gap_km", "sum"),
    ).reset_index()
    state_gap6["gap_pct"] = state_gap6["gap_km"] / state_gap6["highway_km"] * 100
    priority = (
        state_gap6.merge(master[["state", "ev_demand", "n_fast"]], on="state", how="left")
        .sort_values("gap_km", ascending=False).round(1)
    )
    save_table(priority, "state_priority_6_highways.csv")

    # Sensitivity to the distance threshold and fast-charger definition.
    sensitivity = {}
    for threshold in [25, 50, 75, 100]:
        gap = (segments["nearest_fast_km"] > threshold) * segments["length_km"]
        sensitivity[f"{threshold} km"] = 100 * (
            1 - gap.groupby(segments["road_name"]).sum()
            / segments.groupby("road_name")["length_km"].sum()
        )
    sens = pd.DataFrame(sensitivity).round(1)
    save_table(sens.reset_index(names="road_name"), "sensitivity_gap_distance.csv")

    def coverage_by_road(mask):
        points = corridor_chargers.loc[mask]
        distance = gap_distance(segments, points)
        gap = (distance > GAP_KM) * segments["length_km"]
        return 100 * (
            1 - gap.groupby(segments["road_name"]).sum()
            / segments.groupby("road_name")["length_km"].sum()
        )

    definitions = {
        "power >= 22 kW": corridor_chargers["max_power_kw"] >= 22,
        "power >= 25 kW": corridor_chargers["max_power_kw"] >= 25,
        "power >= 50 kW": corridor_chargers["max_power_kw"] >= 50,
        "dataset flag": bool_series(corridor_chargers["has_fast_charging"]),
    }
    sens2 = pd.DataFrame({
        name: coverage_by_road(mask) for name, mask in definitions.items()
    }).round(1)
    save_table(sens2.reset_index(names="road_name"), "sensitivity_fast_definition.csv")

    rating_tbl = sens2.apply(
        lambda series: pd.cut(
            series, [-1, 70, 90, 101], labels=["Poor", "Patchy", "Reliable"]
        ).astype("string")
    )
    core = ["power >= 25 kW", "dataset flag"]
    rating_tbl["verdict"] = np.where(
        rating_tbl[core].nunique(axis=1) == 1,
        "Stable rating", "Depends on definition"
    )
    rating_tbl["coverage_range_%"] = (
        sens2.min(axis=1).round(0).astype(int).astype(str) + " - "
        + sens2.max(axis=1).round(0).astype(int).astype(str)
    )
    save_table(rating_tbl.reset_index(names="road_name"), "highway_rating_by_definition.csv")

    # Add national road-gap metrics to the state master table.
    hwy = seg_nat.groupby("state_key").agg(
        hwy_km=("seg_km", "sum"), gap_km=("gap_km", "sum")
    ).reset_index()
    corr = chargers_gdf.groupby("state_geo")["on_corridor"].sum().rename("n_on_corridor").reset_index()
    corr = corr.rename(columns={"state_geo": "state_key"})
    master = (
        master.drop(columns=["hwy_km", "gap_km", "n_on_corridor"], errors="ignore")
        .merge(hwy, on="state_key", how="left")
        .merge(corr, on="state_key", how="left")
    )
    master[["hwy_km", "gap_km", "n_on_corridor"]] = master[
        ["hwy_km", "gap_km", "n_on_corridor"]
    ].fillna(0)
    master["fast_per_1000_hwy_km"] = np.where(
        master["hwy_km"] > 0, master["n_fast"] / master["hwy_km"] * 1000, np.nan
    )
    master["pct_hwy_gap"] = np.where(
        master["hwy_km"] > 0, master["gap_km"] / master["hwy_km"] * 100, np.nan
    )

    # State gap score and K-means clusters.
    score_data = master.copy()
    median_gap = score_data["pct_hwy_gap"].median()
    score_data["pct_hwy_gap"] = score_data["pct_hwy_gap"].fillna(median_gap)
    score_data["n_demand"] = minmax(np.log1p(score_data["evs_per_charger"]))
    score_data["n_hwygap"] = minmax(score_data["pct_hwy_gap"])
    score_data["n_density"] = minmax(np.log1p(score_data["chargers_per_1000km2"]))
    weights = {"n_demand": 0.40, "n_hwygap": 0.35, "n_density": 0.25}
    score_data[list(weights)] = score_data[list(weights)].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
    score_data["gap_score"] = 100 * sum(
        score_data[column] * weight for column, weight in weights.items()
    )
    score_data["gap_rank"] = score_data["gap_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    score_data["gap_class"] = pd.qcut(
        score_data["gap_score"].rank(method="first"),
        q=3, labels=["Low gap", "Medium gap", "High gap"]
    )
    master = score_data

    feature_columns = ["n_demand", "n_hwygap", "n_density"]
    scaled_features = StandardScaler().fit_transform(master[feature_columns])
    master["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(scaled_features)
    cluster_profile = master.groupby("cluster")[["gap_score", "ev_total", "n_chargers"]].mean().round(1)
    cluster_profile["states"] = master.groupby("cluster").size()
    save_table(cluster_profile.reset_index(), "state_cluster_profile.csv")

    # Test whether the state ranking changes under alternative score weights.
    scenarios = {
        "base 0.40/0.35/0.25": (0.40, 0.35, 0.25),
        "equal weights": (1 / 3, 1 / 3, 1 / 3),
        "demand-heavy": (0.6, 0.2, 0.2),
        "highway-heavy": (0.2, 0.6, 0.2),
        "density-heavy": (0.2, 0.2, 0.6),
    }
    base_top = set(master.nlargest(10, "gap_score")["state"])
    base_rank = master["gap_score"].rank(ascending=False)
    scenario_rows = []
    for name, (a, b, c) in scenarios.items():
        score = 100 * (
            a * master["n_demand"] + b * master["n_hwygap"] + c * master["n_density"]
        )
        scenario_rows.append({
            "scenario": name,
            "top10_overlap": len(base_top & set(master.assign(s=score).nlargest(10, "s")["state"])),
            "rank_correlation": round(base_rank.corr(score.rank(ascending=False), method="spearman"), 3),
        })
    alt_demand = minmax(np.log1p(master["evs_per_fast_charger"]))
    alt_score = 100 * (
        0.40 * alt_demand + 0.35 * master["n_hwygap"] + 0.25 * master["n_density"]
    )
    scenario_rows.append({
        "scenario": "demand = total EV registrations per fast charger",
        "top10_overlap": len(base_top & set(master.assign(s=alt_score).nlargest(10, "s")["state"])),
        "rank_correlation": round(base_rank.corr(alt_score.rank(ascending=False), method="spearman"), 3),
    })
    save_table(pd.DataFrame(scenario_rows), "gap_score_sensitivity.csv")

    save_table(master.drop(columns="geometry").sort_values("gap_score", ascending=False), "state_gap_master.csv")
    save_geojson(master[["state", "state_key", "geometry", "gap_score", "gap_rank", "gap_class", "cluster"]], "state_gap_master.geojson")

    # Store a compact run summary for reporting and visualization.
    top5 = master.nlargest(5, "gap_score")["state"].tolist()
    deserts = master.loc[master["ev_demand"] >= MIN_DEMAND].nlargest(
        3, "evs_per_fast_charger"
    )["state"].tolist()
    zero_chargers = master.loc[master["n_chargers"] == 0, "state"].tolist()
    best = sc.iloc[0]
    worst = sc.iloc[-1]
    longest = sc.sort_values("longest_gap_km", ascending=False).iloc[0]
    summary = pd.DataFrame([
        {"metric": "total_ev_registrations", "value": float(ev_registration["Total"].sum())},
        {"metric": "mapped_operational_chargers", "value": int(len(chargers_gdf))},
        {"metric": "fast_chargers", "value": int(chargers_gdf["is_fast"].sum())},
        {"metric": "national_gap_km", "value": round(national_gap_km, 2)},
        {"metric": "national_gap_pct", "value": round(national_gap_km / seg_nat["seg_km"].sum() * 100, 2)},
        {"metric": "best_covered_target_highway", "value": str(best["road_name"])},
        {"metric": "best_coverage_pct", "value": float(best["covered_pct"])},
        {"metric": "worst_covered_target_highway", "value": str(worst["road_name"])},
        {"metric": "worst_coverage_pct", "value": float(worst["covered_pct"])},
        {"metric": "longest_target_highway_gap_km", "value": float(longest["longest_gap_km"])},
        {"metric": "longest_target_highway_gap_road", "value": str(longest["road_name"])},
        {"metric": "top_gap_score_states", "value": ", ".join(top5)},
        {"metric": "highest_demand_low_fast_supply_states", "value": ", ".join(deserts)},
        {"metric": "states_with_zero_mapped_chargers", "value": ", ".join(zero_chargers) or "none"},
        {"metric": "charger_share_within_corridor_pct", "value": round(float(chargers_gdf["on_corridor"].mean()) * 100, 2)},
    ])
    save_table(summary, "eda_run_summary.csv")

    print("\nKEY FINDINGS")
    for _, row in summary.iterrows():
        print(f"{row['metric']}: {row['value']}")
    print("\nEDA and corridor analysis completed.")
    print("Tables and spatial analysis outputs:", TABLE_DIR)


if __name__ == "__main__":
    run_eda()
