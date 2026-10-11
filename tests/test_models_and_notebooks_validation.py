"""
Validation Test Suite for Models and Notebooks Workflow.
Verifies recent fixes across candidate_models.joblib, cv_benchmark_summary.csv,
and notebooks 01-04 (leakage prevention, key mapping, threshold isolation).
"""

import json
import os
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.evaluation import (
    calculate_lift_and_gain,
    compute_threshold_curve,
    evaluate_holdout_models,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Name mapping used in notebooks/04_Evaluation_and_Diagnostics.ipynb
CV_NAME_MAP = {
    "HistGradientBoosting (Tuned)": "HistGradientBoosting (Tuned)",
    "Random Forest (Tuned)": "Random Forest (Tuned)",
    "Random Forest (Balanced)": "Random Forest",
    "HistGradientBoosting (Balanced)": "HistGradientBoosting",
    "MLP Classifier (Neural)": "MLP Classifier",
    "Logistic Regression (Balanced)": "Logistic Regression",
    "Dummy Baseline": "Dummy Baseline",
}


class TestModelsAndNotebooksValidation(unittest.TestCase):
    """Test suite ensuring all models, preprocessing contracts, and notebook flows remain intact."""

    @classmethod
    def setUpClass(cls):
        cls.models_bundle_path = os.path.join(MODELS_DIR, "candidate_models.joblib")
        cls.cv_summary_path = os.path.join(MODELS_DIR, "cv_benchmark_summary.csv")
        cls.x_train_path = os.path.join(DATA_DIR, "X_train_processed.csv")
        cls.y_train_path = os.path.join(DATA_DIR, "y_train.csv")
        cls.x_test_path = os.path.join(DATA_DIR, "X_test_processed.csv")
        cls.y_test_path = os.path.join(DATA_DIR, "y_test.csv")

        # Load artifacts
        cls.bundle = joblib.load(cls.models_bundle_path)
        cls.candidate_models = cls.bundle["models"]["models"]
        cls.cv_summary = pd.read_csv(cls.cv_summary_path)
        cls.x_test = pd.read_csv(cls.x_test_path)
        cls.y_test = pd.read_csv(cls.y_test_path).squeeze("columns")

    # -------------------------------------------------------------
    # 1. Model Artifacts Integrity
    # -------------------------------------------------------------
    def test_01_models_bundle_structure(self):
        """Verify candidate_models.joblib top-level schema and metadata."""
        self.assertIn("metadata", cls := self.bundle)
        self.assertIn("models", cls)
        self.assertIn("benchmark_summary", cls)

        meta = self.bundle["metadata"]
        self.assertEqual(meta.get("best_model_name"), "HistGradientBoosting (Tuned)")
        self.assertIn("PR-AUC", meta.get("primary_metric", ""))
        self.assertIn("duration", meta.get("leakage_prevention", "").lower())
        self.assertIn("excluded", meta.get("leakage_prevention", "").lower())

    def test_02_all_candidate_models_present_and_callable(self):
        """Verify all 7 candidate models exist and produce calibrated 2-class probabilities."""
        expected_models = [
            "Dummy Baseline",
            "Logistic Regression",
            "Random Forest",
            "HistGradientBoosting",
            "MLP Classifier",
            "HistGradientBoosting (Tuned)",
            "Random Forest (Tuned)",
        ]
        for name in expected_models:
            self.assertIn(name, self.candidate_models, f"Missing candidate model: {name}")
            model = self.candidate_models[name]
            self.assertTrue(hasattr(model, "predict_proba"), f"{name} lacks predict_proba method")

            # Check inference on sample
            sample_x = self.x_test.iloc[:50]
            probs = model.predict_proba(sample_x)
            self.assertEqual(probs.shape, (50, 2), f"{name} output shape mismatch")
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5,
                                       err_msg=f"{name} probabilities do not sum to 1")

    # -------------------------------------------------------------
    # 2. Benchmark Summary & Key Concordance (Fix from commit ab86a46)
    # -------------------------------------------------------------
    def test_03_cv_summary_schema_and_model_mapping(self):
        """Verify cv_benchmark_summary.csv models map without KeyError into candidate_models."""
        self.assertEqual(len(self.cv_summary), 7, "cv_benchmark_summary should have 7 rows")
        expected_cols = [
            "Rank", "Model", "Family", "Val PR-AUC (Mean ± Std)",
            "Val ROC-AUC (Mean ± Std)", "Val Balanced Acc",
            "Val F1-Minority", "Val Brier Score", "Train-Val Gap", "Fit Time (s)"
        ]
        for col in expected_cols:
            self.assertIn(col, self.cv_summary.columns, f"Missing column {col} in benchmark summary")

        # Test name_map concordance
        mapped_keys = self.cv_summary["Model"].map(CV_NAME_MAP)
        self.assertFalse(mapped_keys.isna().any(), "Unmapped model found in CV benchmark summary")
        for key in mapped_keys:
            self.assertIn(key, self.candidate_models, f"Mapped key '{key}' not in candidate_models")

    # -------------------------------------------------------------
    # 3. Data Integrity & Leakage Prevention (Fix from commit 8a656b2)
    # -------------------------------------------------------------
    def test_04_target_leakage_and_shape_invariants(self):
        """Verify duration is excluded and sample sizes match deduplication policy (ADR-003)."""
        x_train = pd.read_csv(self.x_train_path)
        y_train = pd.read_csv(self.y_train_path)

        # Leakage verification
        self.assertNotIn("duration", x_train.columns, "Leakage violation: duration found in X_train")
        self.assertNotIn("duration", self.x_test.columns, "Leakage violation: duration found in X_test")

        # Deduplication sizes: 41,188 - 12 duplicates = 41,176 records (80/20 split)
        self.assertEqual(len(x_train), 32940, f"Expected 32,940 train samples, got {len(x_train)}")
        self.assertEqual(len(self.x_test), 8236, f"Expected 8,236 test samples, got {len(self.x_test)}")
        self.assertEqual(len(y_train), 32940)
        self.assertEqual(len(self.y_test), 8236)
        self.assertEqual(x_train.shape[1], 71, "Feature matrix should contain exactly 71 columns")
        self.assertEqual(self.x_test.shape[1], 71)

        # Null check
        self.assertEqual(x_train.isna().sum().sum(), 0, "Null values detected in X_train")
        self.assertEqual(self.x_test.isna().sum().sum(), 0, "Null values detected in X_test")

    # -------------------------------------------------------------
    # 4. Evaluation Workflow & Metrics
    # -------------------------------------------------------------
    def test_05_holdout_evaluation_metrics_validity(self):
        """Verify evaluate_holdout_models returns valid metrics for all models."""
        holdout_df = evaluate_holdout_models(self.candidate_models, self.x_test, self.y_test)
        self.assertEqual(len(holdout_df), 7)
        self.assertTrue((holdout_df["Holdout PR-AUC"] >= 0.10).all())
        self.assertTrue((holdout_df["Holdout ROC-AUC"] >= 0.50).all())

        # Champion model should achieve strong PR-AUC (> 0.45)
        champion_row = holdout_df[holdout_df["Model"] == "HistGradientBoosting (Tuned)"].iloc[0]
        self.assertGreater(champion_row["Holdout PR-AUC"], 0.45)
        self.assertGreater(champion_row["Holdout ROC-AUC"], 0.80)

    # -------------------------------------------------------------
    # 5. Threshold Optimization Isolation (Fix from commit ab86a46)
    # -------------------------------------------------------------
    def test_06_threshold_tuning_without_test_leakage(self):
        """Verify threshold optimization workflow executes on train-split without test data snooping."""
        x_train = pd.read_csv(self.x_train_path)
        y_train = pd.read_csv(self.y_train_path).squeeze("columns")

        # Reproducible internal split
        x_fit, x_val, y_fit, y_val = train_test_split(
            x_train, y_train, test_size=0.20, random_state=42, stratify=y_train
        )

        champion_model = self.candidate_models["HistGradientBoosting (Tuned)"]
        val_model = clone(champion_model)
        val_model.fit(x_fit, y_fit)
        val_probs = val_model.predict_proba(x_val)[:, 1]

        # Search thresholds on validation split only
        thresholds = np.unique(np.round(np.linspace(0.01, 0.99, 100), 4))
        val_curve = compute_threshold_curve(y_val, val_probs, thresholds=thresholds)

        opt_f1_idx = val_curve["F1"].idxmax()
        frozen_threshold = float(val_curve.loc[opt_f1_idx, "Threshold"])
        self.assertTrue(0.1 <= frozen_threshold <= 0.9, f"Frozen threshold {frozen_threshold} out of bounds")

        # Evaluate frozen threshold on holdout test set
        test_probs = champion_model.predict_proba(self.x_test)[:, 1]
        test_eval = compute_threshold_curve(self.y_test, test_probs, thresholds=np.array([frozen_threshold]))
        test_f1 = float(test_eval.iloc[0]["F1"])
        self.assertGreater(test_f1, 0.40, f"Test F1 at frozen threshold was too low: {test_f1}")

    # -------------------------------------------------------------
    # 6. Notebook JSON Validity
    # -------------------------------------------------------------
    def test_07_notebooks_exist_and_parse_as_valid_json(self):
        """Verify all 4 core notebooks exist, parse properly, and have non-empty cells."""
        notebook_files = [
            "01_Problem_Framing_EDA.ipynb",
            "02_Data_Preprocessing_Pipeline.ipynb",
            "03_Model_Development.ipynb",
            "04_Evaluation_and_Diagnostics.ipynb",
        ]
        for nb_name in notebook_files:
            nb_path = os.path.join(NOTEBOOKS_DIR, nb_name)
            self.assertTrue(os.path.exists(nb_path), f"Notebook missing: {nb_name}")
            with open(nb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("cells", data, f"Notebook {nb_name} missing 'cells'")
            self.assertGreater(len(data["cells"]), 5, f"Notebook {nb_name} has too few cells")


if __name__ == "__main__":
    unittest.main()
