"""
train_regression.py
-------------------
Trains and compares three regression models to predict AQI (numerical).

Models
------
* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

Evaluation metrics
------------------
* MAE  – Mean Absolute Error
* RMSE – Root Mean Squared Error
* R²   – Coefficient of Determination

The best model (highest R²) is saved to models/aqi_regression_model.pkl.
The fitted StandardScaler is saved to models/preprocessor.pkl.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from utils import get_logger, ensure_dir, project_root, rmse

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

MODELS_DIR = project_root() / "models"
CLEAN_DATA_PATH = project_root() / "data" / "cleaned_air_quality.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene",
    "Year", "Month", "Day", "DayOfWeek", "Quarter",
    "City_Encoded",
]


# ── Model definitions ─────────────────────────────────────────────────────────

def get_regression_models() -> Dict[str, object]:
    """Return a dict of {name: unfitted estimator}."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
    }


# ── Training & evaluation ─────────────────────────────────────────────────────

def evaluate_regressor(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Return MAE, RMSE, R² for a fitted model."""
    y_pred = model.predict(X_test)
    return {
        "MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "RMSE": round(rmse(y_test, y_pred), 4),
        "R2": round(float(r2_score(y_test, y_pred)), 4),
    }


def train_regression(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[object, Dict[str, Dict[str, float]], str]:
    """
    Train all regression models and return the best one.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        AQI target.

    Returns
    -------
    best_model : fitted Pipeline
    results    : dict of {model_name: metrics}
    best_name  : name of the best model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Train/test split: %d / %d", len(X_train), len(X_test))

    models = get_regression_models()
    results: Dict[str, Dict[str, float]] = {}
    fitted_models: Dict[str, object] = {}

    for name, estimator in models.items():
        logger.info("Training %s …", name)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        metrics = evaluate_regressor(pipe, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = pipe
        logger.info("  %s → MAE=%.4f  RMSE=%.4f  R²=%.4f", name, metrics["MAE"], metrics["RMSE"], metrics["R2"])

    # Select best by R²
    best_name = max(results, key=lambda n: results[n]["R2"])
    best_model = fitted_models[best_name]
    logger.info("Best regression model: %s  (R²=%.4f)", best_name, results[best_name]["R2"])

    # Persist
    ensure_dir(MODELS_DIR)
    model_path = MODELS_DIR / "aqi_regression_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info("Saved best regression model → %s", model_path)

    # Also save the scaler separately for the preprocessor.pkl slot
    scaler_path = MODELS_DIR / "preprocessor.pkl"
    joblib.dump(best_model.named_steps["scaler"], scaler_path)
    logger.info("Saved preprocessor (scaler) → %s", scaler_path)

    return best_model, results, best_name


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(project_root() / "src"))
    from feature_engineering import engineer_features

    X, y_reg, _ = engineer_features()
    best_model, results, best_name = train_regression(X, y_reg)
    print("\nRegression Results:")
    for name, m in results.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name}: MAE={m['MAE']}  RMSE={m['RMSE']}  R²={m['R2']}{marker}")
