# Notebook Overview

This folder contains `air_quality_analysis.ipynb` — an exploratory analysis notebook used
for data inspection, feature discovery, and visual verification of model behaviour.

Sections:

- 1. Data loading and basic checks — verifies schema and missingness.
- 2. Cleaning steps — demonstrates preprocessing choices before saving the cleaned CSV.
- 3. Feature engineering — shows temporal features and encoding used in modeling.
- 4. EDA visualizations — reproduces key figures saved to `visuals/`.
- 5. Modeling snippets — quick training runs for regression and classification used
  for prototyping (full training happens in `run_pipeline.py`).

Notes:
- Use the repository `requirements.txt` to create a reproducible environment before
  running the notebook.
- Long-running training is intentionally omitted from the notebook to keep it
  responsive during exploration.
