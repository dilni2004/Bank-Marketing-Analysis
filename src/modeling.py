"""
Modeling utilities for Phase 3: Model Development & Cross-Validation Benchmarking.

Author: Nishshanka A. D. N. N. (IT24104095)
Module: IT3091 Machine Learning
Group: 2026-DS-12

Provides modular functions for:
- Leakage-free Stratified 5-Fold Cross-Validation evaluation.
- Imbalance-aware metric computation (PR-AUC, ROC-AUC, Balanced Acc, F1, Brier Score).
- Overfitting diagnostic tracking (train vs validation score gap).
- Benchmark table compilation and visualization helpers.
- Serialized candidate bundle persistence.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def evaluate_cv_model(
    estimator: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    model_name: str,
    family: str,
    cv: Optional[StratifiedKFold] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates an estimator using Stratified 5-Fold Cross Validation.

    Computes both training and validation fold metrics across:
    - PR-AUC (Average Precision) [Primary ranking metric]
    - ROC-AUC
    - Balanced Accuracy
    - F1-Macro
    - F1-Minority (pos_label=1)
    - Brier Score Loss (calibration quality)
    - Wall-clock fit and scoring latency

    Parameters
    ----------
    estimator : BaseEstimator
        Scikit-learn compatible classifier implementing fit and predict_proba or decision_function.
    X : pd.DataFrame or np.ndarray
        Preprocessed training feature matrix (N, D).
    y : pd.Series or np.ndarray
        Binary target vector (N,).
    model_name : str
        Human-readable name of the candidate model.
    family : str
        Algorithmic family (e.g. 'Baseline', 'Linear', 'Bagging', 'Boosting', 'Neural', 'Margin').
    cv : StratifiedKFold, optional
        Pre-configured cross-validator. Defaults to StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    fold_df : pd.DataFrame
        DataFrame with rows corresponding to each fold (0 to 4) with metrics.
    summary : dict
        Aggregated summary containing mean and std for each metric, plus train-val PR-AUC gap.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_vec = y.values.ravel() if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y).ravel()

    fold_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_mat, y_vec)):
        X_train_f, X_val_f = X_mat[train_idx], X_mat[val_idx]
        y_train_f, y_val_f = y_vec[train_idx], y_vec[val_idx]

        clf = clone(estimator)

        # Fit timer
        t_fit_start = time.time()
        clf.fit(X_train_f, y_train_f)
        fit_time = time.time() - t_fit_start

        # Scoring timer
        t_score_start = time.time()
        # Handle probability extraction
        if hasattr(clf, "predict_proba"):
            train_probs = clf.predict_proba(X_train_f)
            val_probs = clf.predict_proba(X_val_f)
            # Handle edge case for single class or binary output
            train_p1 = train_probs[:, 1] if train_probs.shape[1] > 1 else train_probs[:, 0]
            val_p1 = val_probs[:, 1] if val_probs.shape[1] > 1 else val_probs[:, 0]
        elif hasattr(clf, "decision_function"):
            train_scores = clf.decision_function(X_train_f)
            val_scores = clf.decision_function(X_val_f)
            # Min-max / sigmoid mapping for decision function to probability scale
            train_p1 = 1.0 / (1.0 + np.exp(-train_scores))
            val_p1 = 1.0 / (1.0 + np.exp(-val_scores))
        else:
            train_p1 = clf.predict(X_train_f).astype(float)
            val_p1 = clf.predict(X_val_f).astype(float)

        train_preds = clf.predict(X_train_f)
        val_preds = clf.predict(X_val_f)
        score_time = time.time() - t_score_start

        # Compute Metrics
        train_pr_auc = average_precision_score(y_train_f, train_p1)
        val_pr_auc = average_precision_score(y_val_f, val_p1)

        try:
            train_roc_auc = roc_auc_score(y_train_f, train_p1)
            val_roc_auc = roc_auc_score(y_val_f, val_p1)
        except ValueError:
            train_roc_auc = 0.5
            val_roc_auc = 0.5

        train_bacc = balanced_accuracy_score(y_train_f, train_preds)
        val_bacc = balanced_accuracy_score(y_val_f, val_preds)

        train_f1_macro = f1_score(y_train_f, train_preds, average="macro", zero_division=0)
        val_f1_macro = f1_score(y_val_f, val_preds, average="macro", zero_division=0)

        train_f1_pos = f1_score(y_train_f, train_preds, pos_label=1, zero_division=0)
        val_f1_pos = f1_score(y_val_f, val_preds, pos_label=1, zero_division=0)

        train_brier = brier_score_loss(y_train_f, train_p1)
        val_brier = brier_score_loss(y_val_f, val_p1)

        fold_records.append({
            "model_name": model_name,
            "family": family,
            "fold": fold_idx,
            "train_pr_auc": train_pr_auc,
            "val_pr_auc": val_pr_auc,
            "train_roc_auc": train_roc_auc,
            "val_roc_auc": val_roc_auc,
            "train_balanced_acc": train_bacc,
            "val_balanced_acc": val_bacc,
            "train_f1_macro": train_f1_macro,
            "val_f1_macro": val_f1_macro,
            "train_f1_minority": train_f1_pos,
            "val_f1_minority": val_f1_pos,
            "train_brier_score": train_brier,
            "val_brier_score": val_brier,
            "fit_time_sec": fit_time,
            "score_time_sec": score_time,
        })

    fold_df = pd.DataFrame(fold_records)

    summary = {
        "model_name": model_name,
        "family": family,
        "val_pr_auc_mean": float(fold_df["val_pr_auc"].mean()),
        "val_pr_auc_std": float(fold_df["val_pr_auc"].std()),
        "train_pr_auc_mean": float(fold_df["train_pr_auc"].mean()),
        "train_val_pr_auc_gap": float(fold_df["train_pr_auc"].mean() - fold_df["val_pr_auc"].mean()),
        "val_roc_auc_mean": float(fold_df["val_roc_auc"].mean()),
        "val_roc_auc_std": float(fold_df["val_roc_auc"].std()),
        "val_balanced_acc_mean": float(fold_df["val_balanced_acc"].mean()),
        "val_f1_minority_mean": float(fold_df["val_f1_minority"].mean()),
        "val_brier_score_mean": float(fold_df["val_brier_score"].mean()),
        "fit_time_mean_sec": float(fold_df["fit_time_sec"].mean()),
    }

    return fold_df, summary


def compile_cv_benchmark_table(cv_results_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregates fold-level DataFrames into a clean comparison table sorted by validation PR-AUC.

    Parameters
    ----------
    cv_results_list : list of pd.DataFrame
        List of fold evaluation DataFrames produced by evaluate_cv_model.

    Returns
    -------
    summary_table : pd.DataFrame
        Ranked comparison table with mean ± std for key metrics.
    """
    all_folds = pd.concat(cv_results_list, ignore_index=True)
    grouped = all_folds.groupby(["model_name", "family"])

    rows = []
    for (m_name, fam), df_g in grouped:
        val_pr_mean = df_g["val_pr_auc"].mean()
        val_pr_std = df_g["val_pr_auc"].std()
        train_pr_mean = df_g["train_pr_auc"].mean()
        gap = train_pr_mean - val_pr_mean

        val_roc_mean = df_g["val_roc_auc"].mean()
        val_roc_std = df_g["val_roc_auc"].std()

        rows.append({
            "Model": m_name,
            "Family": fam,
            "Val PR-AUC (Mean ± Std)": f"{val_pr_mean:.4f} ± {val_pr_std:.4f}",
            "Val ROC-AUC (Mean ± Std)": f"{val_roc_mean:.4f} ± {val_roc_std:.4f}",
            "Val Balanced Acc": f"{df_g['val_balanced_acc'].mean():.4f}",
            "Val F1-Minority": f"{df_g['val_f1_minority'].mean():.4f}",
            "Val Brier Score": f"{df_g['val_brier_score'].mean():.4f}",
            "Train-Val Gap": f"{gap:.4f}",
            "Fit Time (s)": f"{df_g['fit_time_sec'].mean():.2f}",
            "_raw_val_pr_auc": val_pr_mean,
        })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(by="_raw_val_pr_auc", ascending=False).reset_index(drop=True)
    summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))
    summary_df = summary_df.drop(columns=["_raw_val_pr_auc"])
    return summary_df


def export_candidate_bundle(
    models_dict: Dict[str, Any],
    benchmark_table: pd.DataFrame,
    export_dir: str = "models",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Serializes trained candidate models and summary benchmark metrics to disk.

    Parameters
    ----------
    models_dict : dict
        Mapping of model identifier to fitted estimator / search object.
    benchmark_table : pd.DataFrame
        Ranked performance table.
    export_dir : str, default='models'
        Target output directory.
    metadata : dict, optional
        Additional context (author, phase, git branch, seed).

    Returns
    -------
    saved_paths : dict
        Filepaths of serialized bundle and csv summary table.
    """
    os.makedirs(export_dir, exist_ok=True)
    bundle_path = os.path.join(export_dir, "candidate_models.joblib")
    csv_path = os.path.join(export_dir, "cv_benchmark_summary.csv")

    bundle_payload = {
        "metadata": metadata or {
            "project": "UCI Bank Marketing Campaign Response",
            "phase": "03_Model_Development",
            "author": "Nishshanka A. D. N. N. (IT24104095)",
            "validation_strategy": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "primary_metric": "PR-AUC (average_precision)",
        },
        "models": models_dict,
        "benchmark_summary": benchmark_table.to_dict(orient="records"),
    }

    joblib.dump(bundle_payload, bundle_path)
    benchmark_table.to_csv(csv_path, index=False)

    return {
        "bundle_path": bundle_path,
        "csv_path": csv_path,
    }
