"""
Evaluation, Validation, and Diagnostics Utility Module for Bank Marketing Analysis.
SLIIT IT3091 Machine Learning Group Project (Group ID: 2026-DS-12).
Author: Nawarathna M.S.G. (IT24104206).
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve


def evaluate_holdout_models(
    models_dict: Dict[str, Any],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Evaluates a dictionary of trained scikit-learn estimators on the holdout test set.

    Computes comprehensive ranking and classification metrics:
    PR-AUC (Average Precision), ROC-AUC, Balanced Accuracy, Precision, Recall,
    F1-score, Brier Score Loss, and Log Loss.

    Parameters:
        models_dict: Mapping of model names to fitted estimators.
        X_test: Holdout test feature matrix (must NOT contain post-contact duration).
        y_test: Holdout test ground truth binary labels.
        threshold: Decision threshold for discrete classification metrics (default 0.50).

    Returns:
        pd.DataFrame containing formatted holdout evaluation metrics sorted by PR-AUC descending.
    """
    if isinstance(X_test, pd.DataFrame):
        if "duration" in X_test.columns:
            raise ValueError('Data Leakage Violation: duration found in test features.')

    y_true = np.asarray(y_test).ravel()
    results = []

    for name, model in models_dict.items():
        # Obtain predicted positive probabilities
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            df_vals = model.decision_function(X_test)
            y_probs = 1.0 / (1.0 + np.exp(-df_vals))
        else:
            # Fallback to binary predictions as pseudo-probabilities
            y_probs = model.predict(X_test).astype(float)

        # Discrete predictions at specified decision threshold
        y_pred = (y_probs >= threshold).astype(int)

        pr_auc = float(average_precision_score(y_true, y_probs))
        
        # ROC-AUC calculation (handling edge case if only single class present)
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except ValueError:
            roc_auc = 0.5

        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        brier = float(brier_score_loss(y_true, y_probs))

        # Safe log loss with epsilon clipping
        eps = 1e-15
        y_probs_clipped = np.clip(y_probs, eps, 1.0 - eps)
        loss = float(log_loss(y_true, y_probs_clipped))

        results.append({
            "Model": name,
            "Holdout PR-AUC": round(pr_auc, 4),
            "Holdout ROC-AUC": round(roc_auc, 4),
            "Balanced Acc": round(bal_acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "Brier Score": round(brier, 4),
            "Log Loss": round(loss, 4),
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Holdout PR-AUC", ascending=False).reset_index(drop=True)
    df_results.insert(0, "Rank", range(1, len(df_results) + 1))
    return df_results


def compute_threshold_curve(
    y_true: Union[pd.Series, np.ndarray],
    y_probs: Union[pd.Series, np.ndarray],
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Computes classification trade-off metrics across continuous decision thresholds.

    Parameters:
        y_true: True binary labels.
        y_probs: Predicted positive class probabilities.
        thresholds: Array of thresholds to evaluate in [0, 1] (default np.linspace(0.01, 0.99, 100)).

    Returns:
        pd.DataFrame containing columns:
        [Threshold, Precision, Recall, F1, Balanced Acc, Predicted Positive Rate]
    """
    y_arr = np.asarray(y_true).ravel()
    prob_arr = np.asarray(y_probs).ravel()

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    rows = []
    total_samples = len(y_arr)

    for th in thresholds:
        preds = (prob_arr >= th).astype(int)
        pos_rate = float(preds.sum() / total_samples) if total_samples > 0 else 0.0

        prec = float(precision_score(y_arr, preds, zero_division=0))
        rec = float(recall_score(y_arr, preds, zero_division=0))
        f1 = float(f1_score(y_arr, preds, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_arr, preds))

        rows.append({
            "Threshold": round(float(th), 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "Balanced Acc": round(bal_acc, 4),
            "Predicted Positive Rate": round(pos_rate, 4),
        })

    return pd.DataFrame(rows)


def calculate_lift_and_gain(
    y_true: Union[pd.Series, np.ndarray],
    y_probs: Union[pd.Series, np.ndarray],
    bins: int = 10,
) -> pd.DataFrame:
    """
    Calculates decile cumulative gains, responders captured, hit rate, and lift factor over random targeting.

    Parameters:
        y_true: Ground truth binary target.
        y_probs: Predicted positive probabilities.
        bins: Quantile count (default 10 for deciles).

    Returns:
        pd.DataFrame with decile performance statistics.
    """
    y_arr = np.asarray(y_true).ravel()
    prob_arr = np.asarray(y_probs).ravel()
    n_samples = len(y_arr)
    total_positives = int(y_arr.sum())

    df = pd.DataFrame({"y_true": y_arr, "prob": prob_arr})
    # Sort descending by probability
    df = df.sort_values(by="prob", ascending=False).reset_index(drop=True)

    # Assign equal-sized bins/deciles
    df["decile"] = pd.qcut(df.index, q=bins, labels=False) + 1

    summary_rows = []
    cum_leads = 0
    cum_responders = 0

    for dec in range(1, bins + 1):
        dec_df = df[df["decile"] == dec]
        leads_in_decile = len(dec_df)
        responders_in_decile = int(dec_df["y_true"].sum())

        cum_leads += leads_in_decile
        cum_responders += responders_in_decile

        cum_contact_pct = cum_leads / n_samples
        cum_capture_pct = (cum_responders / total_positives) if total_positives > 0 else 0.0
        hit_rate = (responders_in_decile / leads_in_decile) if leads_in_decile > 0 else 0.0
        lift = (cum_capture_pct / cum_contact_pct) if cum_contact_pct > 0 else 1.0

        summary_rows.append({
            "Decile": dec,
            "Leads": leads_in_decile,
            "Cum Leads": cum_leads,
            "Cum Contact %": round(cum_contact_pct * 100, 2),
            "Responders": responders_in_decile,
            "Cum Responders": cum_responders,
            "Cum Capture %": round(cum_capture_pct * 100, 2),
            "Hit Rate": round(hit_rate, 4),
            "Lift": round(lift, 2),
        })

    return pd.DataFrame(summary_rows)


def compute_calibration_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_probs: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes calibration curve coordinates, Expected Calibration Error (ECE), and Brier Score.

    Parameters:
        y_true: True binary labels.
        y_probs: Predicted probabilities.
        n_bins: Number of bins for reliability diagram.

    Returns:
        dict containing:
        - "prob_true": empirical positive fractions
        - "prob_pred": mean predicted probabilities
        - "brier_score": mean squared probability error
        - "ece": Expected Calibration Error weighted by bin sample counts
    """
    y_arr = np.asarray(y_true).ravel()
    prob_arr = np.clip(np.asarray(y_probs).ravel(), 0.0, 1.0)
    n_samples = len(y_arr)

    # Standard scikit-learn calibration curve
    prob_true, prob_pred = calibration_curve(y_arr, prob_arr, n_bins=n_bins, strategy="uniform")

    # Uniform bin edges for ECE computation
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (prob_arr >= low) & (prob_arr <= high if i == n_bins - 1 else prob_arr < high)
        bin_size = int(mask.sum())
        if bin_size > 0:
            bin_acc = float(y_arr[mask].mean())
            bin_conf = float(prob_arr[mask].mean())
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    brier = float(brier_score_loss(y_arr, prob_arr))

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
    }


def evaluate_subgroup_performance(
    y_true: Union[pd.Series, np.ndarray],
    y_probs: Union[pd.Series, np.ndarray],
    metadata_df: pd.DataFrame,
    attribute_col: str,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Evaluates classification error rates and conversion metrics across demographic subgroups.

    Parameters:
        y_true: True binary target labels.
        y_probs: Predicted probabilities.
        metadata_df: DataFrame containing demographic features aligned with test labels.
        attribute_col: Column name to segment by.
        threshold: Decision threshold for discrete classification.

    Returns:
        pd.DataFrame with error and fairness metrics across cohort categories.
    """
    y_arr = np.asarray(y_true).ravel()
    prob_arr = np.asarray(y_probs).ravel()
    preds = (prob_arr >= threshold).astype(int)

    if attribute_col not in metadata_df.columns:
        raise ValueError(f'Attribute {attribute_col} not found in metadata DataFrame.')

    df_meta = metadata_df.copy().reset_index(drop=True)
    df_meta["y_true"] = y_arr
    df_meta["y_pred"] = preds

    subgroup_metrics = []

    for val, group in df_meta.groupby(attribute_col):
        g_y = group["y_true"].values
        g_pred = group["y_pred"].values
        n_group = len(group)
        positives = int(g_y.sum())
        negatives = n_group - positives
        pred_positives = int(g_pred.sum())

        tn = int(((g_y == 0) & (g_pred == 0)).sum())
        fp = int(((g_y == 0) & (g_pred == 1)).sum())
        fn = int(((g_y == 1) & (g_pred == 0)).sum())
        tp = int(((g_y == 1) & (g_pred == 1)).sum())

        fpr = (fp / negatives) if negatives > 0 else 0.0
        fnr = (fn / positives) if positives > 0 else 0.0
        prec = (tp / pred_positives) if pred_positives > 0 else 0.0
        rec = (tp / positives) if positives > 0 else 0.0
        prevalence = positives / n_group if n_group > 0 else 0.0

        subgroup_metrics.append({
            "Subgroup": str(val),
            "Sample Count": n_group,
            "Prevalence": round(prevalence, 4),
            "Pred Positives": pred_positives,
            "FPR": round(fpr, 4),
            "FNR": round(fnr, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
        })

    res_df = pd.DataFrame(subgroup_metrics)
    return res_df.sort_values(by="Sample Count", ascending=False).reset_index(drop=True)
