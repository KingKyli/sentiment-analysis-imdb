# Project Artifacts

This folder collects the main generated artifacts created during the project upgrade, without moving the core runtime files from their expected locations.

## Included Generated Files

- `metrics.json`
  - copied from `model/metrics.json`
  - evaluation metrics for the final deployable pipeline
- `training_summary.json`
  - copied from `model/training_summary.json`
  - training/export configuration summary for the final pipeline
- `confusion_matrix.png`
  - copied from `docs/assets/confusion_matrix.png`
  - evaluation visualization used in the README and app

## Screenshot Folder

Place README showcase screenshots in `screenshots/` using these names:
- `app_overview.png`
- `app_greek_prediction.png`
- `app_english_prediction.png`
- `app_evaluation.png`

## Core Files Created Or Upgraded

These remain in their runtime locations because moving them would break the project structure:
- `app/app.py`
- `scripts/train_pipeline.py`
- `scripts/evaluate_model.py`
- `model/sentiment_pipeline.pkl`
- `model/metrics.json`
- `model/training_summary.json`
- `docs/assets/confusion_matrix.png`
- `README.md`
- `requirements.txt`
