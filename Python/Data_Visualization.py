import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from Data_Loading import (
    CLEAN_DIR, FIGURE_DIR, TABLE_DIR, load_cleaned_datasets, ensure_output_directories
)

warnings.filterwarnings("ignore", category=FutureWarning)

NAVY_BLUE = "#102A4C"
EMERALD_GREEN = "#00D878"
SKY_BLUE = "#5AA9E6"
LIGHT_BLUE = "#DCEAF5"
WHITE = "#FFFFFF"
RED = "#E5484D"
METRIC_CRS = "EPSG:7755"
FAST_KW = 25
CORRIDOR_KM = 5
GAP_KM = 50
MIN_DEMAND = 5000
TARGET_HIGHWAYS = ["NH-44", "NH-48", "NH-16", "NH-66", "NH-544", "NH-52"]
YEARS = [str(year) for year in range(2017, 2027)]


def finish(name):
    plt.tight_layout()
    path = FIGURE_DIR / f"{name}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved:", path)


def load_table(filename):
    path = TABLE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing EDA output: {path}. Run eda.py before data_visualization.py."
        )
    return pd.read_csv(path)


def main():
    ensure_output_directories()
    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})

    cleaned = load_cleaned_datasets()
    ev_registration = cleaned["ev_registration"].copy()
    chargers_df = cleaned["chargers"].copy()
    states_gdf = cleaned["states"].copy()
    highways_gdf = cleaned["highways"].copy()

    required_outputs = [
        "state_gap_master.csv", "highway_scorecard.csv",
        "state_priority_6_highways.csv", "sensitivity_gap_distance.csv",
        "sensitivity_fast_definition.csv", "national_segments_gap.geojson",
        "segments_6_highways_gap.geojson", "chargers_clean.geojson",
        "fast_corridor_chargers.geojson", "corridor_chargers.geojson",
    ]
    missing = [name for name in required_outputs if not (TABLE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Required EDA outputs are missing: "
            + ", ".join(missing)
            + ". Run eda.py before data_visualization.py."
        )

    # Load analysis tables and spatial outputs created by eda.py.
    master = load_table("state_gap_master.csv")
    if "ev_desert" in master.columns:
        master["ev_desert"] = master["ev_desert"].astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes"]
        )
    sc = load_table("highway_scorecard.csv")
    priority = load_table("state_priority_6_highways.csv")
    sens = load_table("sensitivity_gap_distance.csv").set_index("road_name")
    sens2 = load_table("sensitivity_fast_definition.csv").set_index("road_name")
    rating_tbl = load_table("highway_rating_by_definition.csv")
    scenario_sensitivity = load_table("gap_score_sensitivity.csv")
    cluster_profile = load_table("state_cluster_profile.csv")
    seg_nat = gpd.read_file(TABLE_DIR / "national_segments_gap.geojson").to_crs(METRIC_CRS)
    segments = gpd.read_file(TABLE_DIR / "segments_6_highways_gap.geojson").to_crs(METRIC_CRS)
    chargers_gdf = gpd.read_file(TABLE_DIR / "chargers_clean.geojson").to_crs(METRIC_CRS)
    fast_corridor_chargers = gpd.read_file(TABLE_DIR / "fast_corridor_chargers.geojson").to_crs(METRIC_CRS)
    corridor_chargers = gpd.read_file(TABLE_DIR / "corridor_chargers.geojson").to_crs(METRIC_CRS)
    aliases = {
        "orissa": "odisha",
        "uttaranchal": "uttarakhand",
        "pondicherry": "puducherry",
        "nct of delhi": "delhi",
        "delhi nct": "delhi",
        "ut of dnh and dd": "dadra and nagar haveli and daman and diu",
        "dnh and dd": "dadra and nagar haveli and daman and diu",
        "dadra and nagar haveli": "dadra and nagar haveli and daman and diu",
        "daman and diu": "dadra and nagar haveli and daman and diu",
    }
    states_gdf["state_key"] = (
        states_gdf["state"].astype("string").str.lower()
        .str.replace("&", "and", regex=False)
        .str.replace(r"[^a-z\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip().str.replace("islands", "island", regex=False)
        .replace(aliases)
    )
    states_m = states_gdf.to_crs(METRIC_CRS)

    med_ev = float(master["ev_demand"].median())
    med_fc = float(master["n_fast"].median())

    # 1. National EV registration trend.
    ev_registration.columns = ev_registration.columns.astype(str)
    yearly = ev_registration[YEARS].apply(pd.to_numeric, errors="coerce").sum().rename_axis("year").reset_index(name="registrations")
    yearly["year"] = yearly["year"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 5))
    color = plt.cm.viridis(0.65)
    ax.plot(yearly["year"].iloc[:-1], yearly["registrations"].iloc[:-1], marker="o", color=color, label="All EVs")
    if len(yearly) > 1:
        ax.plot(yearly["year"].iloc[-2:], yearly["registrations"].iloc[-2:], linestyle="--", marker="o", color=color)
    ax.set_title("All EV registrations")
    ax.set_xlabel("Year (2026 may be a partial year)")
    ax.set_ylabel("Registrations")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.legend()
    finish("01_ev_growth")

    # 2. Registration trends for the ten highest-registration states/UTs.
    top10_states = ev_registration.nlargest(10, "Total")
    trend_data = top10_states.melt(
        id_vars="State", value_vars=YEARS, var_name="year", value_name="registrations"
    )
    trend_data["registrations"] = pd.to_numeric(trend_data["registrations"], errors="coerce")
    plt.figure(figsize=(11, 5))
    sns.lineplot(
        data=trend_data, x="year", y="registrations", hue="State",
        hue_order=top10_states["State"].tolist(), palette="viridis", marker="o"
    )
    plt.title("EV registrations by year — top 10 states/UTs")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    finish("02_top10_state_trend")

    # 3. Charger power distribution and fast/slow mix.
    chargers_gdf["max_power_kw"] = pd.to_numeric(chargers_gdf["max_power_kw"], errors="coerce")
    chargers_gdf["is_fast"] = chargers_gdf["is_fast"].astype(str).str.lower().isin(["true", "1", "yes"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(
        chargers_gdf["max_power_kw"].dropna(), bins=30,
        color=plt.cm.viridis(0.65), ax=axes[0]
    )
    axes[0].axvline(FAST_KW, color=plt.cm.viridis(0.95), linestyle="--", label=f"Fast = {FAST_KW} kW+")
    axes[0].legend()
    axes[0].set_title("Charger maximum power (kW)")
    charger_counts = chargers_gdf["is_fast"].map({True: "Fast", False: "Slow/unknown"}).value_counts()
    charger_counts.plot.pie(
        autopct="%1.0f%%", colors=[plt.cm.viridis(0.75), plt.cm.viridis(0.25)], ax=axes[1]
    )
    axes[1].set_ylabel("")
    axes[1].set_title("Fast vs. slow/unknown chargers")
    finish("03_charger_power")

    # 4. Operators with the largest number of mapped chargers.
    operators = chargers_gdf["operator"].fillna("Unknown Operator").value_counts()
    top_operators = operators.head(10)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(top_operators), 1)))
    fig, ax = plt.subplots(figsize=(9, 5))
    top_operators.plot.barh(ax=ax, color=colors, edgecolor=WHITE, width=0.72)
    ax.invert_yaxis()
    ax.set_title("Top 10 EV charging station operators", color=NAVY_BLUE)
    ax.set_xlabel("Number of charging stations", color=NAVY_BLUE)
    ax.set_ylabel("Operator", color=NAVY_BLUE)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, color=LIGHT_BLUE)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    finish("04_top_operators")
    unknown_share = round(operators.get("Unknown Operator", 0) / max(len(chargers_gdf), 1) * 100, 1)
    print("Share with 'Unknown Operator':", unknown_share, "%")

    # 5. Charger power mix in the eight states with the most stations.
    chargers_copy = chargers_gdf.copy()
    bins = [0, 7.5, 22, 50, 1000]
    labels = ["<= 7.5 kW", "7.5–22 kW", "22–50 kW", ">50 kW"]
    chargers_copy["power_band"] = pd.cut(chargers_copy["max_power_kw"], bins=bins, labels=labels)
    top_states = chargers_copy["state_name"].value_counts().head(8).index
    mix = (
        chargers_copy.loc[chargers_copy["state_name"].isin(top_states)]
        .groupby(["state_name", "power_band"], observed=False).size()
        .unstack(fill_value=0)
    )
    mix = mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    mix.loc[top_states[::-1]].plot(kind="barh", stacked=True, figsize=(11, 5), colormap="viridis")
    plt.xlabel("% of chargers")
    plt.ylabel("")
    plt.title("Charger speed mix in the eight states with most chargers")
    plt.legend(title="Maximum power", bbox_to_anchor=(1.01, 1), loc="upper left")
    finish("05_charger_speed_mix")

    # 6. EV-demand versus fast-charger supply quadrant.
    d = master.copy()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(
        d["ev_demand"].clip(lower=1), d["n_fast"] + 1,
        c=np.where(d["ev_desert"].astype(bool), 0.9, 0.35),
        cmap="viridis", vmin=0, vmax=1, s=65, alpha=0.85
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(med_ev, linestyle="--", color=plt.cm.viridis(0.5))
    ax.axhline(med_fc + 1, linestyle="--", color=plt.cm.viridis(0.5))
    for _, row in d.loc[
        d["ev_desert"].astype(bool) | (d["ev_demand"] >= d["ev_demand"].quantile(0.85))
    ].iterrows():
        ax.annotate(
            row["state"], (max(row["ev_demand"], 1), row["n_fast"] + 1),
            fontsize=8, xytext=(4, 4), textcoords="offset points"
        )
    ax.set_xlabel("Total EV registrations (log scale)")
    ax.set_ylabel("Operational fast chargers + 1 (log scale)")
    ax.set_title("EV desert quadrant: demand versus fast-charger supply")
    finish("06_ev_desert_quadrant")

    # 7. States with high EV demand and the highest EVs-per-fast-charger ratio.
    top_ev_deserts = (
        master.loc[master["ev_demand"] >= MIN_DEMAND]
        .nlargest(12, "evs_per_fast_charger").iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(top_ev_deserts), 1)))
    ax.barh(top_ev_deserts["state"], top_ev_deserts["evs_per_fast_charger"], color=colors)
    for y, value in enumerate(top_ev_deserts["evs_per_fast_charger"]):
        ax.text(value, y, f" {value:,.0f}", va="center", fontsize=9)
    ax.set_xlabel("Total EV registrations per operational fast charger (+1 denominator adjustment)")
    ax.set_title(f"Top EV deserts (at least {MIN_DEMAND:,} EV registrations)")
    finish("07_top_ev_deserts")

    # 8. National road network gap map.
    fig, ax = plt.subplots(figsize=(10, 11))
    states_m.boundary.plot(ax=ax, color=LIGHT_BLUE, linewidth=0.5)
    seg_nat.loc[~seg_nat["is_gap"].astype(bool)].plot(ax=ax, color=plt.cm.viridis(0.35), linewidth=0.5)
    seg_nat.loc[seg_nat["is_gap"].astype(bool)].plot(ax=ax, color=plt.cm.viridis(0.95), linewidth=1.2)
    chargers_gdf.loc[chargers_gdf["is_fast"]].plot(ax=ax, color=plt.cm.viridis(0.65), markersize=6)
    ax.set_title(f"National road segments without a fast charger within {GAP_KM} km")
    ax.axis("off")
    finish("08_national_gap_map")

    # 9. Highway coverage scorecard.
    color_map = {"Reliable": 0.75, "Patchy": 0.50, "Poor": 0.15}
    colors = [plt.cm.viridis(color_map.get(str(value), 0.30)) for value in sc["rating"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(sc["road_name"], sc["covered_pct"], color=colors)
    axes[0].axvline(90, linestyle="--", color="grey")
    axes[0].set_xlim(0, 100)
    for y, value in enumerate(sc["covered_pct"]):
        axes[0].text(min(value + 1, 96), y, f"{value:.0f}%", va="center")
    axes[0].invert_yaxis()
    axes[0].set_title(f"% of highway within {GAP_KM} km of a fast charger")
    axes[1].barh(sc["road_name"], sc["longest_gap_km"], color=plt.cm.viridis(0.95))
    axes[1].invert_yaxis()
    axes[1].set_title("Longest continuous charging gap (km)")
    finish("09_highway_scorecard")

    # 10. Coverage and gap map for the six target highways.
    fast_all = chargers_gdf.loc[chargers_gdf["is_fast"]]
    covered_color, gap_color, charger_color = (
        plt.cm.viridis(0.35), plt.cm.viridis(0.95), plt.cm.viridis(0.65)
    )
    fig, ax = plt.subplots(figsize=(10, 12))
    states_m.boundary.plot(ax=ax, color=LIGHT_BLUE, linewidth=1.0)
    segments.loc[~segments["is_gap"].astype(bool)].plot(ax=ax, color=covered_color, linewidth=1.8)
    segments.loc[segments["is_gap"].astype(bool)].plot(ax=ax, color=gap_color, linewidth=3)
    fast_all.plot(ax=ax, color=charger_color, markersize=4, alpha=0.65)
    ax.legend(handles=[
        Line2D([0], [0], color=covered_color, lw=2, label=f"Covered (within {GAP_KM} km)"),
        Line2D([0], [0], color=gap_color, lw=3, label=f"Gap (over {GAP_KM} km)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=charger_color,
               markersize=6, label="Operational fast charger"),
    ], loc="lower left")
    ax.set_axis_off()
    ax.set_title("Fast-charging coverage on six major National Highways")
    finish("10_gap_map_6_highways")

    # 11. One map for each target highway that has segment data.
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for ax, road in zip(axes.ravel(), TARGET_HIGHWAYS):
        road_segments = segments.loc[segments["road_name"] == road].copy()
        road_segments = road_segments.loc[
            road_segments.geometry.notna() & ~road_segments.geometry.is_empty
        ]
        if road_segments.empty:
            ax.set_title(f"{road}: no segment data")
            ax.set_axis_off()
            continue
        states_m.boundary.plot(ax=ax, color=LIGHT_BLUE, linewidth=0.5)
        road_segments.loc[~road_segments["is_gap"].astype(bool)].plot(
            ax=ax, color=covered_color, linewidth=0.5
        )
                # Plot covered and uncovered segments only when data is available.
        covered_segments = road_segments.loc[
            ~road_segments["is_gap"].astype(bool)
        ]

        gap_segments = road_segments.loc[
            road_segments["is_gap"].astype(bool)
        ]

        if not covered_segments.empty:
            covered_segments.plot(
                ax=ax,
                color=covered_color,
                linewidth=0.5
            )

        if not gap_segments.empty:
            gap_segments.plot(
                ax=ax,
                color=gap_color,
                linewidth=2
            )
        road_chargers = fast_corridor_chargers.loc[fast_corridor_chargers["road_name"] == road]
        if not road_chargers.empty:
            road_chargers.plot(ax=ax, color=charger_color, markersize=6)
        minx, miny, maxx, maxy = road_segments.total_bounds
        pad = 150_000
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_axis_off()
        score = sc.loc[sc["road_name"] == road]
        if not score.empty:
            row = score.iloc[0]
            ax.set_title(
                f"{road}: {row['covered_pct']:.0f}% covered, "
                f"longest gap {row['longest_gap_km']:.0f} km"
            )
        else:
            ax.set_title(road)
    finish("11_gap_map_per_highway")

    # 12. States with the largest uncovered distance on target highways.
    p = priority.loc[priority["gap_km"] > 0].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(p), 1)))
    ax.barh(p["state"], p["gap_km"], color=colors)
    for y, (gap, demand) in enumerate(zip(p["gap_km"], p["ev_demand"])):
        ax.text(gap, y, f"  {gap:,.0f} km | {demand:,.0f} EVs", va="center", fontsize=9)
    ax.set_xlabel(f"Highway km with no fast charger within {GAP_KM} km")
    ax.set_title("Priority states for additional corridor charging")
    if not p.empty:
        ax.set_xlim(0, p["gap_km"].max() * 1.45)
    finish("12_state_priority_6_highways")

    # 13. Sensitivity to the definition of a charging gap.
    sens.plot(kind="bar", figsize=(11, 5), colormap="viridis")
    plt.ylabel("% of highway covered")
    plt.xlabel("")
    plt.title("Coverage under different gap-distance thresholds")
    plt.xticks(rotation=0)
    plt.legend(title="Gap threshold")
    finish("13_sensitivity_gap_distance")

    # 14. Sensitivity to the definition of a fast charger.
    sens2.plot(kind="bar", figsize=(11, 5), colormap="viridis")
    plt.ylabel("% of highway covered")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.title("Highway coverage under different fast-charger definitions")
    plt.legend(title="Definition")
    finish("14_sensitivity_fast_definition")

    # 15. State-wise comparison of charger supply and EV registrations.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    chargers_top = master.nlargest(10, "n_chargers")
    colors1 = plt.cm.viridis(np.linspace(0.05, 0.85, len(chargers_top)))
    chargers_top.plot.barh(x="state", y="n_chargers", ax=axes[0], legend=False, color=colors1)
    axes[0].set_title("Top 10 states/UTs: mapped chargers")
    axes[0].invert_yaxis()
    evs_top = master.nlargest(10, "ev_total")
    colors2 = plt.cm.viridis(np.linspace(0.25, 1.0, len(evs_top)))
    evs_top.plot.barh(x="state", y="ev_total", ax=axes[1], legend=False, color=colors2)
    axes[1].set_title("Top 10 states/UTs: EV registrations")
    axes[1].invert_yaxis()
    finish("15_top_states_chargers_vs_evs")

    # 16. Spearman correlations between demand, supply and gap indicators.
    numeric_columns = [
        "ev_total", "ev_cagr_21_25", "n_chargers", "chargers_per_1000km2",
        "evs_per_charger", "n_fast", "pct_hwy_gap"
    ]
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        master[numeric_columns].corr(method="spearman"),
        annot=True, fmt=".2f", cmap="viridis", vmin=-1, vmax=1
    )
    plt.title("Spearman rank correlation")
    finish("16_spearman_heatmap")

    # 17. State clusters by EV registrations and gap score.
    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=master, x="ev_total", y="gap_score", hue="cluster",
        palette="viridis", s=80
    )
    top_cluster_states = master.nlargest(8, "gap_score")
    texts = []
    for _, row in top_cluster_states.iterrows():
        texts.append(plt.text(row["ev_total"], row["gap_score"], row["state"], fontsize=8))
    try:
        from adjustText import adjust_text
        adjust_text(
            texts, x=top_cluster_states["ev_total"], y=top_cluster_states["gap_score"],
            expand=(1.3, 1.5),
            arrowprops={"arrowstyle": "-", "color": "grey", "lw": 0.6},
        )
    except ImportError:
        pass
    plt.xscale("log")
    plt.title("State clusters: EV volume versus gap score")
    finish("17_state_clusters")

    # 18. Choropleth of state-level EV charging gap score.
    choro = states_m.merge(
        master[["state_key", "gap_score", "gap_class"]], on="state_key", how="left"
    )
    fig, ax = plt.subplots(figsize=(9, 10))
    choro.plot(
        column="gap_score", cmap="viridis", legend=True,
        edgecolor="grey", linewidth=0.4,
        missing_kwds={"color": "lightgrey"}, ax=ax
    )
    ax.set_title("EV charging gap score by state/UT (higher = larger gap)")
    ax.axis("off")
    finish("18_gap_score_map")

    # 19. States ranked by the overall gap score.
    top10_gap = master.nlargest(10, "gap_score").iloc[::-1]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(top10_gap)))
    top10_gap.plot.barh(x="state", y="gap_score", figsize=(8, 5), legend=False, color=colors)
    plt.title("Top 10 states/UTs by gap score")
    plt.ylabel("")
    finish("19_top10_gap_score")

    # 20. Compare overall state gap score with the uncovered distance on target highways.
    comparison = master[["state", "gap_score", "ev_total"]].merge(
        priority[["state", "gap_km"]], on="state", how="inner"
    )
    plt.figure(figsize=(9, 6))
    plt.scatter(
        comparison["gap_score"], comparison["gap_km"],
        s=comparison["ev_total"] / max(comparison["ev_total"].max(), 1) * 600 + 30,
        c=comparison["gap_score"], cmap="viridis", alpha=0.75
    )
    texts = []
    for _, row in comparison.iterrows():
        texts.append(plt.text(row["gap_score"], row["gap_km"], row["state"], fontsize=8))
    try:
        from adjustText import adjust_text
        adjust_text(
            texts, x=comparison["gap_score"], y=comparison["gap_km"],
            expand=(1.3, 1.5), force_text=(0.8, 1.2), force_static=(0.5, 0.8),
            arrowprops={"arrowstyle": "-", "color": "grey", "lw": 0.6},
        )
    except ImportError:
        pass
    plt.colorbar(label="State gap score")
    plt.xlabel("State gap score (all roads and chargers)")
    plt.ylabel("Uncovered km on the six target highways")
    plt.title("State gap score versus target-highway gap (bubble = EV registrations)")
    finish("20_score_vs_corridor_gap")

    print("\nVisualization completed.")
    print("Figures saved to:", FIGURE_DIR)


if __name__ == "__main__":
    main()
