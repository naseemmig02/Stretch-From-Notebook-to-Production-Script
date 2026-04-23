"""
Module 5 Week B — Integration Task: Model Comparison & Decision Memo

Module 5 culminating deliverable. Compare 6 model configurations using
5-fold stratified cross-validation, produce PR curves and calibration
plots, log experiments, persist the best model, and demonstrate what
tree-based models capture that linear models cannot.

Run with:  python model_comparison.py
Tests:     pytest tests/ -v
"""

import os
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


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]


# ── Task 1 ────────────────────────────────────────────────────────────────────

def load_and_preprocess(filepath="data/telecom_churn.csv", random_state=42):
    """Load the Petra Telecom dataset and split into train/test sets.

    Uses an 80/20 stratified split. Features are the 8 NUMERIC_FEATURES
    columns. Target is `churned`.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    df = pd.read_csv(filepath)

    X = df[NUMERIC_FEATURES]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ── Task 2 ────────────────────────────────────────────────────────────────────

def define_models():
    """Define 6 model configurations wrapped in sklearn Pipelines.

    LR variants need StandardScaler; tree-based models use 'passthrough'.
    All stochastic models use random_state=42.

    Returns:
        Dict of {name: Pipeline} with 6 entries.
    """
    models = {
        "Dummy": Pipeline([
            ("scaler", "passthrough"),
            ("model", DummyClassifier(strategy="most_frequent", random_state=42)),
        ]),
        "LR_default": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "LR_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            )),
        ]),
        "DT_depth5": Pipeline([
            ("scaler", "passthrough"),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]),
        "RF_default": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )),
        ]),
        "RF_balanced": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight="balanced", random_state=42
            )),
        ]),
    }
    return models


# ── Task 3 ────────────────────────────────────────────────────────────────────

def run_cv_comparison(models, X, y, n_splits=5, random_state=42):
    """Run 5-fold stratified CV on all models.

    Computes mean ± std of accuracy, precision, recall, F1, and PR-AUC.
    PR-AUC uses predict_proba — threshold-independent ranking metric.

    Returns:
        DataFrame with one row per model and columns:
        model, accuracy_mean/std, precision_mean/std, recall_mean/std,
        f1_mean/std, pr_auc_mean/std.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for name, pipeline in models.items():
        fold_scores = {
            "accuracy": [], "precision": [], "recall": [], "f1": [], "pr_auc": []
        }

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)[:, 1]

            fold_scores["accuracy"].append(accuracy_score(y_val, y_pred))
            fold_scores["precision"].append(
                precision_score(y_val, y_pred, zero_division=0)
            )
            fold_scores["recall"].append(
                recall_score(y_val, y_pred, zero_division=0)
            )
            fold_scores["f1"].append(
                f1_score(y_val, y_pred, zero_division=0)
            )
            fold_scores["pr_auc"].append(
                average_precision_score(y_val, y_proba)
            )

        row = {"model": name}
        for metric, scores in fold_scores.items():
            row[f"{metric}_mean"] = np.mean(scores)
            row[f"{metric}_std"] = np.std(scores)
        rows.append(row)

    cols = ["model",
            "accuracy_mean", "accuracy_std",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
            "f1_mean", "f1_std",
            "pr_auc_mean", "pr_auc_std"]
    return pd.DataFrame(rows, columns=cols)


# ── Task 4 ────────────────────────────────────────────────────────────────────

def save_comparison_table(results_df, output_path="results/comparison_table.csv"):
    """Save the comparison DataFrame to CSV."""
    results_df.to_csv(output_path, index=False)
    print(f"  Saved comparison table → {output_path}")


# ── Task 5 ────────────────────────────────────────────────────────────────────

def plot_pr_curves_top3(models, X_test, y_test, output_path="results/pr_curves.png"):
    """Plot PR curves for the top-3 models (by test-set PR-AUC) and save."""
    # Rank all models by PR-AUC on the test set
    pr_aucs = {}
    for name, pipeline in models.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        pr_aucs[name] = average_precision_score(y_test, y_proba)

    top3 = sorted(pr_aucs, key=pr_aucs.get, reverse=True)[:3]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top3:
        PrecisionRecallDisplay.from_estimator(
            models[name], X_test, y_test,
            name=f"{name} (AP={pr_aucs[name]:.3f})",
            ax=ax
        )

    ax.set_title("Precision-Recall Curves — Top 3 Models")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved PR curves → {output_path}")


# ── Task 6 ────────────────────────────────────────────────────────────────────

def plot_calibration_top3(models, X_test, y_test,
                          output_path="results/calibration.png"):
    """Plot calibration curves for the top-3 models and save."""
    # Same top-3 selection logic as Task 5
    pr_aucs = {}
    for name, pipeline in models.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        pr_aucs[name] = average_precision_score(y_test, y_proba)

    top3 = sorted(pr_aucs, key=pr_aucs.get, reverse=True)[:3]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top3:
        CalibrationDisplay.from_estimator(
            models[name], X_test, y_test,
            n_bins=10, name=name, ax=ax
        )

    ax.set_title("Calibration Curves — Top 3 Models")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved calibration plot → {output_path}")


# ── Task 7 ────────────────────────────────────────────────────────────────────

def save_best_model(best_model, output_path="results/best_model.joblib"):
    """Persist the best fitted Pipeline to disk with joblib."""
    dump(best_model, output_path)
    print(f"  Saved best model → {output_path}")


# ── Task 8 ────────────────────────────────────────────────────────────────────

def log_experiment(results_df, output_path="results/experiment_log.csv"):
    """Log all model results with a timestamp column and save to CSV.

    Columns: model_name, accuracy, precision, recall, f1, pr_auc, timestamp.
    """
    timestamp = datetime.now().isoformat()

    log_df = pd.DataFrame({
        "model_name": results_df["model"],
        "accuracy":   results_df["accuracy_mean"],
        "precision":  results_df["precision_mean"],
        "recall":     results_df["recall_mean"],
        "f1":         results_df["f1_mean"],
        "pr_auc":     results_df["pr_auc_mean"],
        "timestamp":  timestamp,
    })

    log_df.to_csv(output_path, index=False)
    print(f"  Saved experiment log → {output_path}")


# ── Task 9 ────────────────────────────────────────────────────────────────────

def find_tree_vs_linear_disagreement(rf_model, lr_model, X_test, y_test,
                                     feature_names, min_diff=0.15):
    """Find the test sample where RF and LR disagree most on P(churn=1).

    Both models are Pipelines so they accept the same raw X_test — the
    Pipeline handles scaling internally for LR.

    Returns:
        Dict with keys: sample_idx, feature_values, rf_proba, lr_proba,
        prob_diff, true_label.  Returns None if no sample meets min_diff.
    """
    rf_probas = rf_model.predict_proba(X_test)[:, 1]
    lr_probas = lr_model.predict_proba(X_test)[:, 1]

    diffs = np.abs(rf_probas - lr_probas)
    max_idx = int(np.argmax(diffs))

    if diffs[max_idx] < min_diff:
        print(f"  No sample with |diff| >= {min_diff}. Max diff: {diffs[max_idx]:.3f}")
        return None

    # X_test may be a DataFrame or ndarray; handle both
    if hasattr(X_test, "iloc"):
        sample_values = X_test.iloc[max_idx].to_dict()
        true_label = int(y_test.iloc[max_idx])
    else:
        sample_values = dict(zip(feature_names, X_test[max_idx]))
        true_label = int(y_test[max_idx])

    return {
        "sample_idx":     max_idx,
        "feature_values": sample_values,
        "rf_proba":       float(rf_probas[max_idx]),
        "lr_proba":       float(lr_probas[max_idx]),
        "prob_diff":      float(diffs[max_idx]),
        "true_label":     true_label,
    }


# ── Orchestrator ──────────────────────────────────────────────────────────────

def main():
    """Orchestrate all 9 integration tasks. Run with: python model_comparison.py"""
    os.makedirs("results", exist_ok=True)

    # Task 1: Load + split
    result = load_and_preprocess()
    if not result:
        print("load_and_preprocess not implemented. Exiting.")
        return
    X_train, X_test, y_train, y_test = result
    print(f"Data: {len(X_train)} train, {len(X_test)} test, "
          f"churn rate: {y_train.mean():.2%}")

    # Task 2: Define models
    models = define_models()
    if not models:
        print("define_models not implemented. Exiting.")
        return
    print(f"\n{len(models)} model configurations defined: {list(models.keys())}")

    # Task 3: Cross-validation comparison
    results_df = run_cv_comparison(models, X_train, y_train)
    if results_df is None:
        print("run_cv_comparison not implemented. Exiting.")
        return
    print("\n=== Model Comparison Table (5-fold CV) ===")
    print(results_df.to_string(index=False))

    # Task 4: Save comparison table
    save_comparison_table(results_df)

    # Fit all models on full training set for plots + persistence
    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

    # Task 5: PR curves (top 3)
    plot_pr_curves_top3(fitted_models, X_test, y_test)

    # Task 6: Calibration plot (top 3)
    plot_calibration_top3(fitted_models, X_test, y_test)

    # Task 7: Save best model
    best_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    print(f"\nBest model by PR-AUC: {best_name}")
    save_best_model(fitted_models[best_name])

    # Task 8: Experiment log
    log_experiment(results_df)

    # Task 9: Tree-vs-linear disagreement
    rf_pipeline = fitted_models["RF_default"]
    lr_pipeline = fitted_models["LR_default"]
    disagreement = find_tree_vs_linear_disagreement(
        rf_pipeline, lr_pipeline, X_test, y_test, NUMERIC_FEATURES
    )
    if disagreement:
        print(f"\n--- Tree-vs-linear disagreement (sample idx={disagreement['sample_idx']}) ---")
        print(f"  RF P(churn=1)={disagreement['rf_proba']:.3f}  "
              f"LR P(churn=1)={disagreement['lr_proba']:.3f}")
        print(f"  |diff| = {disagreement['prob_diff']:.3f}   "
              f"true label = {disagreement['true_label']}")

        md_lines = [
            "# Tree vs. Linear Disagreement Analysis",
            "",
            "## Sample Details",
            "",
            f"- **Test-set index:** {disagreement['sample_idx']}",
            f"- **True label:** {disagreement['true_label']}",
            f"- **RF predicted P(churn=1):** {disagreement['rf_proba']:.4f}",
            f"- **LR predicted P(churn=1):** {disagreement['lr_proba']:.4f}",
            f"- **Probability difference:** {disagreement['prob_diff']:.4f}",
            "",
            "## Feature Values",
            "",
        ]
        for feat, val in disagreement["feature_values"].items():
            md_lines.append(f"- **{feat}:** {val}")
        md_lines.extend([
            "",
            "## Structural Explanation",
            "",
            "<!-- Write 2-3 sentences explaining WHY these models disagree on this",
            "     sample. Point to a specific feature interaction, non-monotonic",
            "     relationship, or threshold effect the tree captured that the",
            "     linear model could not. -->",
            "",
        ])
        with open("results/tree_vs_linear_disagreement.md", "w") as f:
            f.write("\n".join(md_lines))
        print("  Saved to results/tree_vs_linear_disagreement.md")

    print("\n--- All results saved to results/ ---")
    print("Write your decision memo in the PR description (Task 10).")


if __name__ == "__main__":
    main()