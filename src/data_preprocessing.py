"""
data_preprocessing.py
---------------------
Loads, cleans, and saves the raw air-quality dataset.

Steps
-----
1. Load data/city_day.csv
2. Parse Date column as datetime
3. Drop rows where AQI or AQI_Bucket is missing
4. Impute remaining numeric NaNs with column medians
5. Remove duplicate rows
6. Save cleaned file to data/cleaned_air_quality.csv
"""

from pathlib import Path

import pandas as pd

from utils import get_logger, ensure_dir, project_root

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

RAW_DATA_PATH = project_root() / "data" / "city_day.csv"
CLEAN_DATA_PATH = project_root() / "data" / "cleaned_air_quality.csv"

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene",
]


# ── Core functions ────────────────────────────────────────────────────────────

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and parse the Date column."""
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path, parse_dates=["Date"])
    logger.info("Raw shape: %s", df.shape)
    return df


def drop_missing_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where AQI or AQI_Bucket is NaN."""
    before = len(df)
    df = df.dropna(subset=["AQI", "AQI_Bucket"])
    after = len(df)
    logger.info("Dropped %d rows with missing AQI/AQI_Bucket (%d → %d)", before - after, before, after)
    return df.reset_index(drop=True)


def impute_numeric_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with column medians."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    missing_before = df[numeric_cols].isna().sum().sum()
    medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(medians)
    missing_after = df[numeric_cols].isna().sum().sum()
    logger.info(
        "Imputed %d missing numeric values with column medians (remaining: %d)",
        missing_before - missing_after,
        missing_after,
    )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info("Removed %d duplicate rows (%d → %d)", before - after, before, after)
    return df.reset_index(drop=True)


def preprocess(
    raw_path: Path = RAW_DATA_PATH,
    clean_path: Path = CLEAN_DATA_PATH,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    raw_path : Path
        Location of the raw CSV file.
    clean_path : Path
        Destination for the cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = load_raw_data(raw_path)
    df = drop_missing_targets(df)
    df = impute_numeric_medians(df)
    df = remove_duplicates(df)

    ensure_dir(clean_path.parent)
    df.to_csv(clean_path, index=False)
    logger.info("Cleaned dataset saved to %s  (shape: %s)", clean_path, df.shape)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preprocess()
