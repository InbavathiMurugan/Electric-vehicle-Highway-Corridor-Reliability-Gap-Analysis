""" EV Highway Corridor Reliability Gap Analysis."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
RAW_DIR = DATASET_DIR / "Raw datasets"
CLEAN_DIR = DATASET_DIR / "Cleaned datasets"
OUTPUT_DIR = PROJECT_ROOT / "Visualization"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

RAW_FILES = {
    "ev_registration": RAW_DIR / "All_type_of_EV_registration.csv",
    "chargers": RAW_DIR / "ev_charging_stations.csv",
    "states": RAW_DIR / "States_and_Union_Territories.parquet",
    "highways": RAW_DIR / "National_Highways.parquet",
}

CLEAN_FILES = {
    "ev_registration": CLEAN_DIR / "Ev_registration_cleaned.csv",
    "chargers": CLEAN_DIR / "Ev_chargers_cleaned.csv",
    "states": CLEAN_DIR / "States_and_Union_Territories_Cleaned.parquet",
    "highways": CLEAN_DIR / "highways_clean.geojson",
}


def _check_files(file_map):
    missing = [str(path) for path in file_map.values() if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required dataset file(s) were not found:\n"
            f"{formatted}\n"
            f"Project root detected as: {PROJECT_ROOT}"
        )


def load_raw_datasets():
    """Load the four source datasets used by data_cleaning.py."""
    _check_files(RAW_FILES)
    return {
        "ev_registration": pd.read_csv(RAW_FILES["ev_registration"]),
        "chargers": pd.read_csv(RAW_FILES["chargers"]),
        "states": gpd.read_parquet(RAW_FILES["states"]),
        "highways": gpd.read_parquet(RAW_FILES["highways"]),
    }


def load_cleaned_datasets():
    """Load the four cleaned datasets for EDA and analysis."""
    _check_files(CLEAN_FILES)
    ev_registration = pd.read_csv(CLEAN_FILES["ev_registration"])
    ev_registration.columns = ev_registration.columns.astype(str)

    chargers = pd.read_csv(CLEAN_FILES["chargers"])
    states = gpd.read_parquet(CLEAN_FILES["states"])
    highways = gpd.read_file(CLEAN_FILES["highways"])

    return {
        "ev_registration": ev_registration,
        "chargers": chargers,
        "states": states,
        "highways": highways,
    }


def ensure_output_directories():
    """Create the standard project output folders."""
    for folder in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    raw = load_raw_datasets()
    print("\nRAW DATASET SUMMARY")
    for name, frame in raw.items():
        print(f"{name:16s} shape={frame.shape}, columns={len(frame.columns)}")

    if all(path.exists() for path in CLEAN_FILES.values()):
        clean = load_cleaned_datasets()
        print("\nCLEANED DATASET SUMMARY")
        for name, frame in clean.items():
            print(f"{name:16s} shape={frame.shape}, columns={len(frame.columns)}")
    else:
        print("\nCleaned datasets do not exist yet. Run data_cleaning.py next.")
