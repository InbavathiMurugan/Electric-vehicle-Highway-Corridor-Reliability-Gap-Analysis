import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry

try:
    from shapely.validation import make_valid
except ImportError:
    from shapely import make_valid

from Data_Loading import (
    RAW_DIR, CLEAN_DIR, load_raw_datasets, ensure_output_directories
)

METRIC_CRS = "EPSG:7755"
SOURCE_CRS = "EPSG:4326"
EXPECTED_TOTAL = 36
EXPECTED_STATES = 28
EXPECTED_UTS = 8
YEARS = [str(year) for year in range(2017, 2027)]


def clean_ev_registration(raw):
    """Apply the EV registration cleaning and validation steps from the notebook."""
    ev_df = raw.copy()
    print("\nEV REGISTRATION")
    print("Original shape:", ev_df.shape)

    if "State" not in ev_df.columns:
        ev_df = ev_df.rename(columns={ev_df.columns[0]: "State"})

    ev_df["State"] = ev_df["State"].astype("string").str.strip()
    ev_df = ev_df.loc[ev_df["State"].str.casefold() != "total"].copy()

    missing_years = [year for year in YEARS if year not in ev_df.columns]
    if missing_years:
        raise ValueError(f"EV registration dataset is missing year columns: {missing_years}")
    if "Total" not in ev_df.columns:
        raise ValueError("EV registration dataset is missing the 'Total' column.")

    for column in YEARS + ["Total"]:
        ev_df[column] = pd.to_numeric(ev_df[column], errors="coerce")

    print("Shape after removing Total row:", ev_df.shape)
    print("Duplicate state names:", int(ev_df["State"].duplicated().sum()))
    print("Missing values:\n", ev_df.isna().sum().to_string())
    print("Negative values:\n", (ev_df[YEARS + ["Total"]] < 0).sum().to_string())
    calculated_total = ev_df[YEARS].sum(axis=1)
    print("Rows where Total differs from yearly sum:", int((ev_df["Total"] != calculated_total).sum()))
    print("Duplicate complete rows:", int(ev_df.duplicated().sum()))
    print("Unique states/UTs:", int(ev_df["State"].nunique()))

    # Keep the notebook's state/year structure and export a clean, typed table.
    ev_df = ev_df.reset_index(drop=True)
    print("Final shape:", ev_df.shape)
    return ev_df


def clean_chargers(raw):
    """Apply the notebook's station, state, coordinate and power cleaning."""
    chargers = raw.copy()
    print("\nEV CHARGING STATIONS")
    print("Original shape:", chargers.shape)

    required = [
        "ID", "name", "latitude", "longitude", "town", "state", "postcode",
        "operator", "status", "is_operational", "max_power_kw", "total_points",
        "has_fast_charging", "date_created", "date_last_verified"
    ]
    missing = [column for column in required if column not in chargers.columns]
    if missing:
        raise ValueError(
            "The raw charging-station CSV is missing required notebook fields: "
            f"{missing}. Use the raw ev_charging_stations.csv created from the API response."
        )

    chargers["state"] = chargers["state"].astype("string").str.strip()
    state_name_map = {
        "KARNATAKA": "Karnataka", "karnataka": "Karnataka", "Karnatak": "Karnataka",
        "TAMILNADU": "Tamil Nadu", "TamilNadu": "Tamil Nadu",
        "Tamilnadu": "Tamil Nadu", "Tamil nadu": "Tamil Nadu",
        "UTTAR PRADESH": "Uttar Pradesh", "uttar pradesh": "Uttar Pradesh",
        "rajasthan": "Rajasthan", "west bengal": "West Bengal",
        "GJ": "Gujarat", "KA": "Karnataka", "MH": "Maharashtra",
        "MP": "Madhya Pradesh", "RJ": "Rajasthan",
        "Maharshtra": "Maharashtra", "Mahrashtra": "Maharashtra",
        "Keraka": "Kerala", "Keral": "Kerala", "Lerala": "Kerala",
        "Tamil Nasdu": "Tamil Nadu", "Uttarakhnad": "Uttarakhand",
        "New Dehi": "Delhi", "Ahmedabad": "Gujarat",
        "Bangalore Urban": "Karnataka", "Burari": "Delhi",
        "Villupuram": "Tamil Nadu", "chennai": "Tamil Nadu",
        "India": None
    }
    chargers["state"] = chargers["state"].replace(state_name_map)
    chargers["state"] = chargers["state"].astype("string").str.strip().str.title()
    chargers["state"] = chargers["state"].replace({"None": pd.NA, "Nan": pd.NA, "": pd.NA})

    before = len(chargers)
    chargers = chargers.dropna(subset=["state"]).copy()
    print("Rows removed for missing/invalid state:", before - len(chargers))

    chargers["town"] = chargers["town"].fillna("Unknown").astype("string").str.strip().str.title()
    chargers["postcode"] = chargers["postcode"].fillna("Unknown")
    chargers["operator"] = chargers["operator"].fillna("Unknown Operator")
    chargers["operator"] = chargers["operator"].replace({"(Unknown Operator)": "Unknown Operator"})

    chargers["latitude"] = pd.to_numeric(chargers["latitude"], errors="coerce")
    chargers["longitude"] = pd.to_numeric(chargers["longitude"], errors="coerce")
    chargers["max_power_kw"] = pd.to_numeric(chargers["max_power_kw"], errors="coerce").fillna(0)
    chargers["total_points"] = pd.to_numeric(chargers["total_points"], errors="coerce").fillna(0)
    def parse_bool(series):
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        if pd.api.types.is_numeric_dtype(series):
            return series.fillna(0).astype(float).ne(0)
        return series.astype("string").str.strip().str.lower().isin(
            ["true", "1", "yes", "y", "operational"]
        )

    chargers["is_operational"] = parse_bool(chargers["is_operational"])
    chargers["has_fast_charging"] = parse_bool(chargers["has_fast_charging"])

    before = len(chargers)
    chargers = chargers.loc[chargers["max_power_kw"] <= 500].copy()
    print("Rows removed for power above 500 kW:", before - len(chargers))

    before = len(chargers)
    chargers = chargers.drop_duplicates(subset=["latitude", "longitude"]).copy()
    print("Duplicate locations removed:", before - len(chargers))

    valid_coords = (
        chargers["latitude"].between(6, 38)
        & chargers["longitude"].between(68, 98)
    )
    invalid_coords = ~valid_coords
    print("Invalid coordinates identified:", int(invalid_coords.sum()))

    # Preserve the explicitly reviewed exclusions from the notebook.
    confirmed_invalid_ids = [510821, 301542, 479330]
    before = len(chargers)
    chargers = chargers.loc[~chargers["ID"].isin(confirmed_invalid_ids)].copy()
    print("Notebook-reviewed invalid IDs removed:", before - len(chargers))

    # Revalidate after the explicit exclusions; do not silently retain bad map points.
    valid_coords = (
        chargers["latitude"].between(6, 38)
        & chargers["longitude"].between(68, 98)
    )
    before = len(chargers)
    chargers = chargers.loc[valid_coords].copy()
    print("Remaining invalid coordinate rows removed:", before - len(chargers))

    chargers["date_created"] = pd.to_datetime(chargers["date_created"], errors="coerce")
    chargers["date_last_verified"] = pd.to_datetime(chargers["date_last_verified"], errors="coerce")

    chargers = chargers[required].copy().reset_index(drop=True)
    print("Final shape:", chargers.shape)
    print("Operational stations:", int(chargers["is_operational"].sum()))
    print("Fast-charging flag count:", int(chargers["has_fast_charging"].sum()))
    print("Missing values:\n", chargers.isna().sum().to_string())
    return chargers


def decode_geometry(value):
    """Decode geometry values stored as Shapely objects or WKB."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, BaseGeometry):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return wkb.loads(bytes(value))
    if isinstance(value, str):
        value = value.strip()
        return wkb.loads(value, hex=True) if value else None
    if pd.isna(value):
        return None
    raise TypeError(f"Unsupported geometry value type: {type(value)}")


def _polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        parts = []
        for part in geom.geoms:
            parts.extend(_polygon_parts(part))
        return parts
    return []


def clean_states(raw):
    """Clean state/UT names, codes, polygon geometries and areas."""
    print("\nSTATE AND UNION TERRITORY BOUNDARIES")
    print("Raw shape:", raw.shape)
    required = ["STNAME", "STCODE11", "State_LGD", "geometry"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"State boundary dataset is missing columns: {missing}")

    states = raw[required].copy().rename(columns={
        "STNAME": "state", "STCODE11": "state_code", "State_LGD": "lgd_code"
    })
    states["geometry"] = states["geometry"].apply(decode_geometry)
    states = gpd.GeoDataFrame(states, geometry="geometry", crs=SOURCE_CRS)

    states["state"] = (
        states["state"].astype("string").str.strip()
        .str.replace("&", "and", regex=False)
        .str.replace(",", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.title().str.replace(r"\bAnd\b", "and", regex=True).str.strip()
    )
    state_name_mapping = {
        "andaman and nicobar": "Andaman and Nicobar Islands",
        "andaman and nicobar islands": "Andaman and Nicobar Islands",
        "dadra nagar haveli daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
        "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
        "nct of delhi": "Delhi",
        "national capital territory of delhi": "Delhi",
        "jammu and kashmir": "Jammu and Kashmir",
    }
    states["state"] = states["state"].str.casefold().map(state_name_mapping).fillna(states["state"])
    states["state_code"] = states["state_code"].astype("string").str.strip()
    states["lgd_code"] = pd.to_numeric(states["lgd_code"], errors="raise").astype("Int64")
    states["state_key"] = states["state"].str.casefold()

    states_list = {
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
        "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
        "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
    }
    union_territories = {
        "Andaman and Nicobar Islands", "Chandigarh",
        "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir",
        "Ladakh", "Lakshadweep", "Puducherry"
    }
    states["admin_type"] = states["state"].map(
        lambda name: "State" if name in states_list
        else "Union Territory" if name in union_territories
        else "Unknown"
    )
    unknown = states.loc[states["admin_type"] == "Unknown", "state"].tolist()
    if unknown:
        raise ValueError(f"Unclassified administrative units: {unknown}")

    states = states.loc[states.geometry.notna() & ~states.geometry.is_empty].copy()
    invalid_mask = ~states.geometry.is_valid
    if invalid_mask.any():
        states.loc[invalid_mask, "geometry"] = states.loc[invalid_mask, "geometry"].apply(make_valid)
    states["geometry"] = states.geometry.apply(
        lambda geom: (
            parts[0] if len(parts) == 1 else MultiPolygon(parts)
        ) if (parts := _polygon_parts(geom)) else None
    )
    states = states.loc[states.geometry.notna() & ~states.geometry.is_empty].copy()
    states = gpd.GeoDataFrame(states, geometry="geometry", crs=SOURCE_CRS)

    if len(states) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} state/UT polygons; found {len(states)}.")
    if states["state"].duplicated().any():
        raise ValueError("Duplicate state/UT names found.")
    if states["state_code"].isna().any() or states["lgd_code"].isna().any():
        raise ValueError("Missing state codes or LGD codes found.")
    if not states.geometry.is_valid.all():
        raise ValueError("Invalid state/UT geometries remain after repair.")
    if not states.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Non-polygon geometries remain in the state/UT dataset.")

    projected = states.to_crs(METRIC_CRS)
    if projected.crs is None or not projected.crs.is_projected:
        raise ValueError(f"Expected a projected CRS; received {projected.crs}.")
    area_km2 = projected.geometry.area / 1_000_000
    if area_km2.isna().any() or (area_km2 <= 0).any() or np.isinf(area_km2).any():
        raise ValueError("State/UT polygon areas contain invalid values.")
    states["area_km2"] = area_km2.round(6)
    states["n_parts"] = states.geometry.apply(
        lambda geom: len(geom.geoms) if geom.geom_type == "MultiPolygon" else 1
    )

    assert len(states) == EXPECTED_TOTAL
    assert states["state"].nunique() == EXPECTED_TOTAL
    assert states["state_code"].is_unique
    assert states["lgd_code"].is_unique
    assert states["state_key"].is_unique
    assert states.geometry.notna().all() and states.geometry.is_valid.all()
    assert (states["area_km2"] > 0).all()
    print("Final shape:", states.shape)
    print("Administrative types:\n", states["admin_type"].value_counts().to_string())
    print("Geometry types:\n", states.geom_type.value_counts().to_string())
    return states


def clean_highways(raw):
    """Clean the highway geometries and calculate their lengths in kilometres."""
    print("\nNATIONAL HIGHWAYS")
    print("Raw shape:", raw.shape)
    required = ["state_ut", "road_name", "road_type", "status", "geometry"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Highway dataset is missing columns: {missing}")

    highways = raw[required].copy()
    if highways.crs is None:
        raise ValueError("Highway source CRS is missing.")

    for column in ["state_ut", "road_name", "road_type", "status"]:
        highways[column] = highways[column].astype("string").str.strip()

    highways["road_name"] = highways["road_name"].replace({"": pd.NA, "Delete": pd.NA})
    highways["state_ut"] = highways["state_ut"].replace({
        "MADHYA PRADESH": "Madhya Pradesh",
        "MADHYAPRADESH": "Madhya Pradesh",
        "Daman Diu": "Daman and Diu",
    })

    missing_geometry = int(highways.geometry.isna().sum())
    highways = highways.loc[highways.geometry.notna()].copy()
    empty_geometry = int(highways.geometry.is_empty.sum())
    highways = highways.loc[~highways.geometry.is_empty].copy()

    invalid_before = int((~highways.geometry.is_valid).sum())
    if invalid_before:
        invalid_mask = ~highways.geometry.is_valid
        highways.loc[invalid_mask, "geometry"] = highways.loc[invalid_mask, "geometry"].make_valid()

    line_types = ["LineString", "MultiLineString"]
    non_line_mask = ~highways.geometry.geom_type.isin(line_types)
    non_line_count = int(non_line_mask.sum())
    highways = highways.loc[~non_line_mask].copy()

    before_dedup = len(highways)
    highways = highways.drop_duplicates().reset_index(drop=True)
    print("Missing geometries removed:", missing_geometry)
    print("Empty geometries removed:", empty_geometry)
    print("Invalid geometries before repair:", invalid_before)
    print("Non-line geometries excluded:", non_line_count)
    print("Duplicate records removed:", before_dedup - len(highways))

    metric = highways.to_crs(METRIC_CRS)
    highways["length_km"] = metric.geometry.length / 1000
    if (~highways.geometry.is_valid).any() or highways.geometry.is_empty.any():
        raise ValueError("Invalid or empty highway geometries remain.")
    if highways["length_km"].isna().any() or (highways["length_km"] <= 0).any():
        raise ValueError("Highway lengths contain missing or non-positive values.")

    print("Final shape:", highways.shape)
    print("CRS:", highways.crs)
    print("Road types:\n", highways["road_type"].value_counts(dropna=False).head(10).to_string())
    print("Length summary (km):\n", highways["length_km"].describe().to_string())
    return highways


def main():
    ensure_output_directories()
    raw = load_raw_datasets()
    for name, frame in raw.items():
        print(f"Loaded {name}: shape={frame.shape}, columns={len(frame.columns)}")

    cleaned = {
        "ev_registration": clean_ev_registration(raw["ev_registration"]),
        "chargers": clean_chargers(raw["chargers"]),
        "states": clean_states(raw["states"]),
        "highways": clean_highways(raw["highways"]),
    }

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    cleaned["ev_registration"].to_csv(CLEAN_DIR / "Ev_registration_cleaned.csv", index=False)
    cleaned["chargers"].to_csv(CLEAN_DIR / "Ev_chargers_cleaned.csv", index=False)
    cleaned["states"].to_parquet(CLEAN_DIR / "States_and_Union_Territories_Cleaned.parquet", index=False)
    cleaned["highways"].to_crs("EPSG:4326").to_file(
        CLEAN_DIR / "highways_clean.geojson", driver="GeoJSON"
    )

    print("\nCLEANING SUMMARY")
    for name, frame in cleaned.items():
        print(f"{name:16s} final shape={frame.shape}")
    print("Cleaned datasets saved to:", CLEAN_DIR)


if __name__ == "__main__":
    main()
