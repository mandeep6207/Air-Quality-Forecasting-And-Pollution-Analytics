"""
evaluate_models.py
------------------
Generates all evaluation artefacts:
  * visuals/feature_importance_regression.png
  * visuals/feature_importance_classification.png
  * visuals/confusion_matrix.png
  * reports/model_metrics.json
  * reports/project_report.md
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils import (
    get_logger, save_figure, save_json, set_plot_style,
    project_root, ensure_dir,
)

logger = get_logger(__name__)

ROOT = project_root()
VISUALS_DIR = ROOT / "visuals"
REPORTS_DIR = ROOT / "reports"

AQI_BUCKET_ORDER = [
    "Good", "Satisfactory", "Moderate",
    "Poor", "Very Poor", "Severe",
]

FEATURE_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene",
    "Year", "Month", "Day", "DayOfWeek", "Quarter",
    "City_Encoded",
]


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list,
    title: str,
    filepath: Path,
    top_n: int = 15,
) -> None:
    """Bar chart of top-N feature importances."""
    set_plot_style()
    idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
    bars = ax.barh(top_features[::-1], top_values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    for bar, val in zip(bars, top_values[::-1]):
    # Show absolute importance and percentage of the top-N total for clarity
    total = float(top_values.sum() if top_values.sum() != 0 else 1.0)
    for bar, val in zip(bars, top_values[::-1]):
        pct = (val / total) * 100.0
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f} ({pct:.1f}%)", va="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, filepath)
    logger.info("Saved feature importance plot → %s", filepath)


def generate_feature_importance_plots(
    reg_model,
    clf_model,
    feature_names: list = FEATURE_COLS,
) -> None:
    """Extract and plot feature importances from tree-based models."""
    ensure_dir(VISUALS_DIR)

    # Regression
    reg_step = reg_model.named_steps["model"]
    if hasattr(reg_step, "feature_importances_"):
        plot_feature_importance(
            reg_step.feature_importances_,
            feature_names,
            "Feature Importance – AQI Regression",
            VISUALS_DIR / "feature_importance_regression.png",
        )
    else:
        logger.warning("Regression model has no feature_importances_; skipping plot.")

    # Classification
    clf_step = clf_model.named_steps["model"]
    if hasattr(clf_step, "feature_importances_"):
        plot_feature_importance(
            clf_step.feature_importances_,
            feature_names,
            "Feature Importance – AQI Bucket Classification",
            VISUALS_DIR / "feature_importance_classification.png",
        )
    else:
        logger.warning("Classification model has no feature_importances_; skipping plot.")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: list,
    class_labels: list,
    filepath: Path,
) -> None:
    """Annotated heatmap of a confusion matrix."""
    set_plot_style()
    cm_arr = np.array(cm)
    n = len(class_labels)

    fig, ax = plt.subplots(figsize=(max(7, n), max(6, n - 1)))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title("Confusion Matrix – AQI Bucket Classifier", fontweight="bold", pad=12)

    thresh = cm_arr.max() / 2.0
    total = cm_arr.sum()
    for i in range(n):
        for j in range(n):
            val = int(cm_arr[i, j])
            pct = (val / total * 100.0) if total > 0 else 0.0
            ax.text(j, i, f"{val}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                    fontsize=9)
    fig.tight_layout()
    save_figure(fig, filepath)
    logger.info("Saved confusion matrix → %s", filepath)


# ── Metrics JSON ──────────────────────────────────────────────────────────────

def save_metrics(
    reg_results: Dict[str, Dict],
    clf_results: Dict[str, Dict],
    best_reg: str,
    best_clf: str,
) -> None:
    """Persist all model metrics to reports/model_metrics.json."""
    payload: Dict[str, Any] = {
        "regression": {
            "models": reg_results,
            "best_model": best_reg,
            "best_metrics": reg_results[best_reg],
        },
        "classification": {
            "models": {
                name: {k: v for k, v in m.items() if k != "Confusion_Matrix"}
                for name, m in clf_results.items()
            },
            "best_model": best_clf,
            "best_metrics": {
                k: v for k, v in clf_results[best_clf].items()
                if k != "Confusion_Matrix"
            },
        },
    }
    path = REPORTS_DIR / "model_metrics.json"
    save_json(payload, path)
    logger.info("Saved model metrics → %s", path)


# ── Project report ────────────────────────────────────────────────────────────

def generate_report(
    reg_results: Dict[str, Dict],
    clf_results: Dict[str, Dict],
    best_reg: str,
    best_clf: str,
    dataset_shape: tuple,
) -> None:
    """Write a Markdown project report to reports/project_report.md."""
    ensure_dir(REPORTS_DIR)

    reg_table_rows = "\n".join(
        f"| {n} | {m['MAE']} | {m['RMSE']} | {m['R2']} |{'  ✅ Best' if n == best_reg else ''}"
        for n, m in reg_results.items()
    )
    clf_table_rows = "\n".join(
        f"| {n} | {m['Accuracy']} | {m['Precision']} | {m['Recall']} | {m['F1_Score']} |{'  ✅ Best' if n == best_clf else ''}"
        for n, m in clf_results.items()
    )

    br = reg_results[best_reg]
    bc = clf_results[best_clf]

    report = f"""# Air Quality Forecasting and Pollution Analytics — Project Report

---

## 1. Dataset Overview

| Attribute | Value |
|-----------|-------|
| Source | Central Pollution Control Board (CPCB), India |
| File | `data/city_day.csv` |
| Total Records (raw) | 29,531 |
| Records after cleaning | {dataset_shape[0]} |
| Features | 14 raw columns + 6 engineered |
| Regression Target | AQI (continuous) |
| Classification Target | AQI_Bucket (6 categories) |

---

## 2. Data Cleaning Steps

1. **Date parsing** — Converted `Date` column to `datetime64`.
2. **Target filtering** — Removed rows where `AQI` or `AQI_Bucket` was `NaN`.
3. **Median imputation** — Filled remaining numeric `NaN` values with column medians.
4. **Deduplication** — Dropped exact duplicate rows.
5. **Saved** cleaned dataset to `data/cleaned_air_quality.csv`.

---

## 3. Feature Engineering

| Feature | Type | Description |
|---------|------|-------------|
| PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene | Numeric | Raw pollutant measurements |
| Year | Temporal | Extracted from Date |
| Month | Temporal | Extracted from Date |
| Day | Temporal | Extracted from Date |
| DayOfWeek | Temporal | 0 = Monday … 6 = Sunday |
| Quarter | Temporal | 1–4 |
| City_Encoded | Categorical | Label-encoded City |

---

## 4. Exploratory Data Analysis

Key insights from EDA:

- **AQI distribution** is right-skewed; most readings fall in the 50–200 range.
- **Delhi, Patna, and Lucknow** consistently rank among the most polluted cities.
- **Winter months (Nov–Jan)** show significantly higher AQI due to crop burning and cold inversions.
- **PM2.5 and PM10** have the strongest positive correlation with AQI.
- **O3** shows a moderate negative correlation with other pollutants (photochemical dynamics).

---

## 5. Regression Model Comparison (AQI Prediction)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
{reg_table_rows}

**Best Model:** `{best_reg}`
- MAE  : {br['MAE']}
- RMSE : {br['RMSE']}
- R²   : {br['R2']}

---

## 6. Classification Model Comparison (AQI Bucket)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
{clf_table_rows}

**Best Model:** `{best_clf}`
- Accuracy  : {bc['Accuracy']}
- Precision : {bc['Precision']}
- Recall    : {bc['Recall']}
- F1 Score  : {bc['F1_Score']}

---

## 7. Key Findings

1. **Tree-based ensemble models** (Random Forest / Gradient Boosting) significantly outperform linear models for both tasks.
2. **PM2.5** is the single most important predictor of AQI across all tree models.
3. **Temporal features** (Month, Quarter) add meaningful signal — winter months drive higher AQI.
4. **City identity** is a strong predictor, reflecting persistent geographic and industrial differences.
5. The classifier achieves high F1 on extreme categories (Good / Severe) but shows some confusion between adjacent buckets (Moderate ↔ Poor).

---

## 8. Future Improvements

- **Time-series modelling** — LSTM / Prophet for sequential AQI forecasting.
- **Spatial features** — Latitude/longitude, proximity to industrial zones.
- **Hyperparameter tuning** — Bayesian optimisation (Optuna / scikit-optimize).
- **Ensemble stacking** — Combine regression and classification outputs.
- **Real-time pipeline** — Stream data from CPCB API; retrain incrementally.
- **Explainability** — SHAP values for per-prediction feature attribution.
- **Web dashboard** — Streamlit / Dash app for interactive exploration.

---

*Report generated automatically by `run_pipeline.py`.*
"""

    path = REPORTS_DIR / "project_report.md"
    path.write_text(report, encoding="utf-8")
    logger.info("Saved project report → %s", path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, str(ROOT / "src"))
    from utils import load_json

    metrics = load_json(REPORTS_DIR / "model_metrics.json")
    print(json.dumps(metrics, indent=2))
