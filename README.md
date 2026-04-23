# Telecom Churn Model Comparison Pipeline

A production-quality CLI tool to compare multiple machine learning models for predicting telecom customer churn.

## Installation

1. Clone the repository.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The script `compare_models.py` runs a full model comparison pipeline, including cross-validation, performance metrics, and visualization.

### Arguments

| Argument | Type | Description |
| :--- | :--- | :--- |
| `--data-path` | Required | Path to the input dataset (CSV format). |
| `--output-dir` | Optional | Directory where results and plots are saved (default: `./output`). |
| `--n-folds` | Optional | Number of cross-validation folds (default: 5). |
| `--random-seed` | Optional | Random seed for reproducibility (default: 42). |
| `--dry-run` | Flag | Validates the data and prints configuration without training models. |
| `--verbose` | Flag | Enables detailed debug logging. |

### Example Commands

#### 1. Normal Run
Run the full pipeline with default settings:
```bash
python compare_models.py --data-path data/telecom_churn.csv --output-dir ./results
```

#### 2. Dry Run
Validate the data and configuration before committing to a full training run:
```bash
python compare_models.py --data-path data/telecom_churn.csv --dry-run
```

## Outputs

The script saves the following artifacts to the specified `--output-dir`:
- `comparison_table.csv`: Mean and standard deviation of metrics for all models.
- `pr_curves.png`: Precision-Recall curves for the top 3 models.
- `calibration.png`: Calibration curves for the top 3 models.
- `best_model.joblib`: The serialized best-performing model pipeline.
- `experiment_log.csv`: A log of the run with timestamps and performance summaries.
