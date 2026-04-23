import argparse
import logging
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (PrecisionRecallDisplay, average_precision_score,
                             precision_score, recall_score,
                             f1_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Configure constants
NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]
TARGET_COLUMN = "churned"

def setup_logging():
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_data(data_path):
    """Load the dataset from a CSV file."""
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    logger.info(f"Loading data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

def validate_data(df):
    """Validate the dataset for required columns and report distributions."""
    logger.info("Validating data...")
    missing_cols = [col for col in NUMERIC_FEATURES + [TARGET_COLUMN] if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Check for nulls in required columns
    null_counts = df[NUMERIC_FEATURES + [TARGET_COLUMN]].isnull().sum()
    if null_counts.any():
        logger.warning(f"Found null values in required columns:\n{null_counts[null_counts > 0]}")
    
    # Report class distribution
    dist = df[TARGET_COLUMN].value_counts(normalize=True)
    logger.info(f"Target distribution ('{TARGET_COLUMN}'):")
    for val, count in dist.items():
        logger.info(f"  Class {val}: {count:.2%}")
    
    return True

def define_models(random_seed=42):
    """Define model configurations wrapped in sklearn Pipelines."""
    logger.info(f"Defining model pipelines (seed: {random_seed})")
    models = {
        "Dummy": Pipeline([
            ("scaler", "passthrough"),
            ("model", DummyClassifier(strategy="most_frequent", random_state=random_seed)),
        ]),
        "LR_default": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_seed)),
        ]),
        "LR_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=random_seed
            )),
        ]),
        "DT_depth5": Pipeline([
            ("scaler", "passthrough"),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=random_seed)),
        ]),
        "RF_default": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=random_seed
            )),
        ]),
        "RF_balanced": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight="balanced", random_state=random_seed
            )),
        ]),
    }
    return models

def train_and_evaluate(models, X, y, n_folds=5, random_seed=42):
    """Run cross-validation on all models."""
    logger.info(f"Starting {n_folds}-fold cross-validation (seed: {random_seed})")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    rows = []

    for name, pipeline in models.items():
        logger.info(f"Evaluating {name}...")
        fold_scores = {
            "accuracy": [], "precision": [], "recall": [], "f1": [], "pr_auc": []
        }

        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)[:, 1]

            fold_scores["accuracy"].append(accuracy_score(y_val, y_pred))
            fold_scores["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            fold_scores["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            fold_scores["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            fold_scores["pr_auc"].append(average_precision_score(y_val, y_proba))
            
            logger.debug(f"  {name} - Fold {i+1} PR-AUC: {fold_scores['pr_auc'][-1]:.4f}")

        row = {"model": name}
        for metric, scores in fold_scores.items():
            row[f"{metric}_mean"] = np.mean(scores)
            row[f"{metric}_std"] = np.std(scores)
        rows.append(row)
        logger.info(f"  {name} Mean PR-AUC: {row['pr_auc_mean']:.4f}")

    cols = ["model",
            "accuracy_mean", "accuracy_std",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
            "f1_mean", "f1_std",
            "pr_auc_mean", "pr_auc_std"]
    return pd.DataFrame(rows, columns=cols)

def save_results(results_df, models, X_train, y_train, X_test, y_test, output_dir):
    """Save metrics, plots, and best model to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving all results to {output_dir}")

    # 1. Comparison table
    results_path = os.path.join(output_dir, "comparison_table.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved comparison table to {results_path}")

    # 2. Fit models on full training data for evaluation on test set
    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

    # 3. PR curves (top 3)
    pr_aucs = {name: average_precision_score(y_test, pipeline.predict_proba(X_test)[:, 1])
               for name, pipeline in fitted_models.items()}
    top3 = sorted(pr_aucs, key=pr_aucs.get, reverse=True)[:3]

    pr_path = os.path.join(output_dir, "pr_curves.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top3:
        PrecisionRecallDisplay.from_estimator(
            fitted_models[name], X_test, y_test,
            name=f"{name} (AP={pr_aucs[name]:.3f})",
            ax=ax
        )
    ax.set_title("Precision-Recall Curves — Top 3 Models")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved PR curves plot to {pr_path}")

    # 4. Calibration curves (top 3)
    cal_path = os.path.join(output_dir, "calibration.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top3:
        CalibrationDisplay.from_estimator(
            fitted_models[name], X_test, y_test,
            n_bins=10, name=name, ax=ax
        )
    ax.set_title("Calibration Curves — Top 3 Models")
    fig.tight_layout()
    fig.savefig(cal_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved calibration plot to {cal_path}")

    # 5. Best model persistence
    best_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    dump(fitted_models[best_name], best_model_path)
    logger.info(f"Saved best model ({best_name}) to {best_model_path}")

    # 6. Experiment log
    log_path = os.path.join(output_dir, "experiment_log.csv")
    log_df = pd.DataFrame({
        "model_name": results_df["model"],
        "accuracy":   results_df["accuracy_mean"],
        "precision":  results_df["precision_mean"],
        "recall":     results_df["recall_mean"],
        "f1":         results_df["f1_mean"],
        "pr_auc":     results_df["pr_auc_mean"],
        "timestamp":  datetime.now().isoformat(),
    })
    log_df.to_csv(log_path, index=False)
    logger.info(f"Saved experiment log to {log_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run a full model comparison pipeline for telecom churn prediction."
    )
    parser.add_argument("--data-path", required=True, help="Path to the input dataset (CSV)")
    parser.add_argument("--output-dir", default="./output", help="Directory where results and plots are saved (default: ./output)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Validates the data and prints the pipeline configuration without training any models")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 1. Load Data
    df = load_data(args.data_path)

    # 2. Validate Data
    validate_data(df)

    # Prepare features and target
    X = df[NUMERIC_FEATURES]
    y = df[TARGET_COLUMN]

    # 3. Dry Run Check
    if args.dry_run:
        logger.info("--- DRY RUN MODE ---")
        logger.info(f"Pipeline Configuration:")
        logger.info(f"  Data Path: {args.data_path}")
        logger.info(f"  Output Directory: {args.output_dir}")
        logger.info(f"  CV Folds: {args.n_folds}")
        logger.info(f"  Random Seed: {args.random_seed}")
        logger.info(f"  Features: {NUMERIC_FEATURES}")
        logger.info(f"  Target: {TARGET_COLUMN}")
        
        models = define_models(args.random_seed)
        logger.info(f"  Models to Compare: {list(models.keys())}")
        logger.info("Validation successful. No models will be trained.")
        return

    # 4. Train-Test Split for final evaluation/plots
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.random_seed
    )
    logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")

    # 5. Define Models
    models = define_models(args.random_seed)

    # 6. Run CV Comparison
    results_df = train_and_evaluate(models, X_train, y_train, args.n_folds, args.random_seed)

    # 7. Save Results
    save_results(results_df, models, X_train, y_train, X_test, y_test, args.output_dir)
    
    logger.info("Pipeline complete.")

if __name__ == "__main__":
    main()
