"""
data_preprocessing.py
---------------------
Small, well-documented preprocessing utilities for the project.

This module provides a compact, reproducible preprocessing pipeline used by
the project's notebook and pipeline runner. It focuses on defensive data
handling and produces a clean CSV suitable for feature engineering and
training. The functions are small and testable, and log helpful diagnostic
messages about rows removed and values imputed.

Key steps performed by ``preprocess()``:
- Load the raw CSV and parse the ``Date`` column as datetime
- Drop rows missing target values (``AQI`` or ``AQI_Bucket``)
- Impute numeric columns using column medians (no inplace surprises)
- Remove exact duplicate rows
- Persist the cleaned CSV to a stable location

I/O
---
- Input: the raw CSV located at ``data/city_day.csv`` by default.
- Output: cleaned CSV written to ``data/cleaned_air_quality.csv`` (overwrites).
- Side outputs: none by default; callers may persist metadata separately.

Design notes
------------
The pipeline is intentionally conservative: imputation uses medians to avoid
introducing bias from extreme values, and duplicate removal only drops exact
row copies. These choices make the output predictable and robust for
downstream modelling.

"""

from pathlib import Path

import pandas as pd

from utils import get_logger, ensure_dir, project_root, save_json

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
    # Restrict imputation to known pollutant numeric columns to avoid
    # unexpectedly overwriting unrelated numeric fields (e.g., IDs).
    numeric_cols = [c for c in POLLUTANT_COLS if c in df.columns]
    if not numeric_cols:
        logger.warning("No pollutant numeric columns found for imputation; skipping.")
        return df

    missing_before = df[numeric_cols].isna().sum().sum()
    medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(medians)
    missing_after = df[numeric_cols].isna().sum().sum()
    logger.info(
        "Imputed %d missing pollutant values using medians (remaining: %d)",
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
    # Notes
    # -----
    # - The function writes `clean_path` with `index=False` to keep the CSV
    #   portable and avoid leaking dataframe indices into downstream steps.
    # - Only numeric columns are imputed (non-numeric columns are left intact).
    # - This function is safe to call multiple times; it will overwrite the
    #   cleaned CSV each run and log actions taken.
    #
    # Example
    # -------
    # >>> from pathlib import Path
    # >>> df = preprocess(raw_path=Path('data/city_day.csv'))
    # >>> df.shape
    # (29531, 25)
    df = load_raw_data(raw_path)
    df = drop_missing_targets(df)
    df = impute_numeric_medians(df)
    df = remove_duplicates(df)

    ensure_dir(clean_path.parent)
    df.to_csv(clean_path, index=False)
    logger.info("Cleaned dataset saved to %s  (shape: %s)", clean_path, df.shape)

    # Also persist a tiny metadata file describing the cleaned export (useful
    # for downstream reporting and quick checks without loading the full CSV).
    meta = {
        "clean_path": str(clean_path),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }
    save_json(meta, project_root() / "reports" / "cleaned_metadata.json")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preprocess()
