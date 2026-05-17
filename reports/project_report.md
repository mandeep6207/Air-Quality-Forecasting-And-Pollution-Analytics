# Air Quality Forecasting and Pollution Analytics — Project Report

---

## 1. Dataset Overview

| Attribute | Value |
|-----------|-------|
| Source | Central Pollution Control Board (CPCB), India |
| File | `data/city_day.csv` |
| Total Records (raw) | 29,531 |
| Records after cleaning | 24850 |
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
| Linear Regression | 30.9888 | 59.2599 | 0.8082 |
| Random Forest Regressor | 20.7132 | 40.6391 | 0.9098 |
| Gradient Boosting Regressor | 20.9055 | 39.6147 | 0.9143 |  ✅ Best

**Best Model:** `Gradient Boosting Regressor`
- MAE  : 20.9055
- RMSE : 39.6147
- R²   : 0.9143

---

## 6. Classification Model Comparison (AQI Bucket)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.7531 | 0.7509 | 0.7531 | 0.7456 |
| Random Forest Classifier | 0.8165 | 0.8159 | 0.8165 | 0.8143 |  ✅ Best
| Gradient Boosting Classifier | 0.807 | 0.8059 | 0.807 | 0.8062 |

**Best Model:** `Random Forest Classifier`
- Accuracy  : 0.8165
- Precision : 0.8159
- Recall    : 0.8165
- F1 Score  : 0.8143

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
