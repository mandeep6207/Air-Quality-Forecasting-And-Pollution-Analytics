"""
run_pipeline.py
---------------
Master pipeline script for the Air Quality Forecasting project.

Usage
-----
    python run_pipeline.py

What it does
------------
1.  Preprocesses raw data  → data/cleaned_air_quality.csv
2.  Engineers features     → X, y_reg, y_clf
3.  Runs EDA               → visuals/*.png
4.  Trains regression      → models/aqi_regression_model.pkl
5.  Trains classification  → models/aqi_bucket_classifier.pkl
6.  Generates eval plots   → visuals/feature_importance_*.png
                             visuals/confusion_matrix.png
7.  Saves metrics          → reports/model_metrics.json
8.  Writes project report  → reports/project_report.md
"""

import sys
import time
from pathlib import Path

# ── Make src/ importable ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from utils import get_logger, project_root

logger = get_logger("pipeline")


def main() -> None:
    start = time.time()
    logger.info("=" * 60)
    logger.info("  Air Quality Forecasting — Full Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    logger.info("\n[1/7] Data Preprocessing")
    from data_preprocessing import preprocess
    df_clean = preprocess()

    # ── Step 2: Feature Engineering ───────────────────────────────────────────
    logger.info("\n[2/7] Feature Engineering")
    from feature_engineering import build_feature_matrix
    X, y_reg, y_clf = build_feature_matrix(df_clean)

    # ── Step 3: EDA Visualizations ────────────────────────────────────────────
    logger.info("\n[3/7] EDA Visualizations")
    from eda_visualizations import run_eda
    run_eda()

    # ── Step 4: Regression Training ───────────────────────────────────────────
    logger.info("\n[4/7] Regression Model Training")
    from train_regression import train_regression
    best_reg_model, reg_results, best_reg_name = train_regression(X, y_reg)

    # ── Step 5: Classification Training ──────────────────────────────────────
    logger.info("\n[5/7] Classification Model Training")
    from train_classification import train_classification
    best_clf_model, clf_results, best_clf_name, le, y_test_enc = train_classification(X, y_clf)

    # ── Step 6: Evaluation Plots ──────────────────────────────────────────────
    logger.info("\n[6/7] Generating Evaluation Plots")
    from evaluate_models import (
        generate_feature_importance_plots,
        plot_confusion_matrix,
        save_metrics,
        generate_report,
    )
    from pathlib import Path as _Path

    generate_feature_importance_plots(best_reg_model, best_clf_model)

    # Confusion matrix — use encoded labels from best classifier
    best_clf_metrics = clf_results[best_clf_name]
    cm = best_clf_metrics["Confusion_Matrix"]
    # Build label list from LabelEncoder
    class_labels = list(le.classes_)
    plot_confusion_matrix(
        cm,
        class_labels,
        ROOT / "visuals" / "confusion_matrix.png",
    )

    # ── Step 7: Reports ───────────────────────────────────────────────────────
    logger.info("\n[7/7] Saving Metrics & Report")
    save_metrics(reg_results, clf_results, best_reg_name, best_clf_name)
    generate_report(
        reg_results,
        clf_results,
        best_reg_name,
        best_clf_name,
        dataset_shape=df_clean.shape,
    )

    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info("  Pipeline complete in %.1f seconds", elapsed)
    logger.info("=" * 60)
    logger.info("\nOutputs:")
    logger.info("  Cleaned data  → data/cleaned_air_quality.csv")
    logger.info("  Models        → models/")
    logger.info("  Visuals       → visuals/")
    logger.info("  Reports       → reports/")


if __name__ == "__main__":
    main()
