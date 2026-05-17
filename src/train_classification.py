"""
train_classification.py
-----------------------
Trains and compares three classification models to predict AQI_Bucket.

Models
------
* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier

Evaluation metrics
------------------
* Accuracy
* Precision (weighted)
* Recall    (weighted)
* F1 Score  (weighted)
* Confusion Matrix

The best model (highest F1) is saved to models/aqi_bucket_classifier.pkl.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from utils import get_logger, ensure_dir, project_root

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

MODELS_DIR = project_root() / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ── Model definitions ─────────────────────────────────────────────────────────

def get_classification_models() -> Dict[str, object]:
    """Return a dict of {name: unfitted estimator}."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
    }


# ── Training & evaluation ─────────────────────────────────────────────────────

def evaluate_classifier(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    """Return accuracy, precision, recall, F1, and confusion matrix."""
    y_pred = model.predict(X_test)
    return {
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "F1_Score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_classification(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[object, Dict[str, Dict], str, object, pd.Series]:
    """
    Train all classifiers and return the best one.

    Returns
    -------
    best_model  : fitted Pipeline
    results     : dict of {model_name: metrics}
    best_name   : name of the best model
    le          : fitted LabelEncoder for AQI_Bucket
    y_test      : held-out labels (for confusion matrix plot)
    """
    # Encode string labels
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), name=y.name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )
    logger.info("Train/test split: %d / %d", len(X_train), len(X_test))

    models = get_classification_models()
    results: Dict[str, Dict] = {}
    fitted_models: Dict[str, object] = {}

    for name, estimator in models.items():
        logger.info("Training %s …", name)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        metrics = evaluate_classifier(pipe, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = pipe
        logger.info(
            "  %s → Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f",
            name, metrics["Accuracy"], metrics["Precision"],
            metrics["Recall"], metrics["F1_Score"],
        )

    # Select best by F1
    best_name = max(results, key=lambda n: results[n]["F1_Score"])
    best_model = fitted_models[best_name]
    logger.info("Best classifier: %s  (F1=%.4f)", best_name, results[best_name]["F1_Score"])

    # Persist
    ensure_dir(MODELS_DIR)
    clf_path = MODELS_DIR / "aqi_bucket_classifier.pkl"
    joblib.dump(best_model, clf_path)
    logger.info("Saved best classifier → %s", clf_path)

    return best_model, results, best_name, le, y_test


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(project_root() / "src"))
    from feature_engineering import engineer_features

    X, _, y_clf = engineer_features()
    best_model, results, best_name, le, y_test = train_classification(X, y_clf)
    print("\nClassification Results:")
    for name, m in results.items():
        marker = " ← BEST" if name == best_name else ""
        print(
            f"  {name}: Acc={m['Accuracy']}  P={m['Precision']}  "
            f"R={m['Recall']}  F1={m['F1_Score']}{marker}"
        )
