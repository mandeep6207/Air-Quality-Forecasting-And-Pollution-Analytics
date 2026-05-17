"""
feature_engineering.py
-----------------------
Builds model-ready feature matrices from the cleaned dataset.

Features created
----------------
Temporal  : Year, Month, Day, DayOfWeek, Quarter
Categorical: City  → integer label-encoded as City_Encoded
Pollutants : PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3,
             Benzene, Toluene, Xylene  (already numeric)
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils import get_logger, project_root

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CLEAN_DATA_PATH = project_root() / "data" / "cleaned_air_quality.csv"

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene",
]

TEMPORAL_COLS = ["Year", "Month", "Day", "DayOfWeek", "Quarter", "DayOfYear", "IsWeekend"]

FEATURE_COLS = POLLUTANT_COLS + TEMPORAL_COLS + ["City_Encoded"]

REGRESSION_TARGET = "AQI"
CLASSIFICATION_TARGET = "AQI_Bucket"


# ── Core functions ────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Year, Month, Day, DayOfWeek, Quarter from the Date column."""
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek   # 0 = Monday
    df["Quarter"] = df["Date"].dt.quarter
    # Additional temporal signals for stronger seasonality capture
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsWeekend"] = df["Date"].dt.dayofweek >= 5
    logger.info("Temporal features added: %s", TEMPORAL_COLS)
    return df


def encode_city(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Label-encode the City column.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with new City_Encoded column.
    le : LabelEncoder
        Fitted encoder (needed for inverse-transform / persistence).
    """
    df = df.copy()
    le = LabelEncoder()
    df["City_Encoded"] = le.fit_transform(df["City"].astype(str))
    logger.info("City encoded: %d unique cities", len(le.classes_))
    return df, le


def build_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Apply all feature engineering steps and return X, y_reg, y_clf.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of data_preprocessing.preprocess).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y_reg : pd.Series
        Regression target (AQI).
    y_clf : pd.Series
        Classification target (AQI_Bucket).
    """
    df = add_temporal_features(df)
    df, _ = encode_city(df)

    X = df[FEATURE_COLS].copy()
    y_reg = df[REGRESSION_TARGET].copy()
    y_clf = df[CLASSIFICATION_TARGET].copy()

    logger.info(
        "Feature matrix built: X=%s  |  y_reg=%s  |  y_clf=%s",
        X.shape, y_reg.shape, y_clf.shape,
    )
    return X, y_reg, y_clf


def engineer_features(clean_path: Path = CLEAN_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load cleaned data and return engineered (X, y_reg, y_clf).

    Convenience wrapper used by the pipeline runner.
    """
    df = pd.read_csv(clean_path, parse_dates=["Date"])
    return build_feature_matrix(df)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y_reg, y_clf = engineer_features()
    print("X shape      :", X.shape)
    print("Features     :", list(X.columns))
    print("AQI range    :", y_reg.min(), "–", y_reg.max())
    print("AQI buckets  :", y_clf.unique())
