"""
Unit and Contract Tests for Evaluation Utilities (src/evaluation.py).
Phase 4: Evaluation, Validation, and Diagnostics.
Author: Nawarathna M.S.G. (IT24104206).
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation import (
    evaluate_holdout_models,
    compute_threshold_curve,
    calculate_lift_and_gain,
    compute_calibration_metrics,
    evaluate_subgroup_performance,
)


class TestEvaluationContract(unittest.TestCase):
    """Unit test suite for src/evaluation.py."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n_samples = 300
        cls.n_features = 8
        cls.X_test = pd.DataFrame(
            np.random.randn(cls.n_samples, cls.n_features),
            columns=[f"feat_{i}" for i in range(cls.n_features)]
        )
        # 12% prevalence positive labels
        cls.y_test = pd.Series(
            (np.random.rand(cls.n_samples) < 0.12).astype(int),
            name="y"
        )
        # Synthetic fitted models
        cls.dummy = DummyClassifier(strategy="prior").fit(cls.X_test, cls.y_test)
        cls.logreg = LogisticRegression(random_state=42).fit(cls.X_test, cls.y_test)
        cls.models_dict = {
            "Dummy Baseline": cls.dummy,
            "Logistic Regression": cls.logreg,
        }

    def test_01_evaluate_holdout_models(self):
        """Verify evaluate_holdout_models computes metrics and catches leakage."""
        results = evaluate_holdout_models(self.models_dict, self.X_test, self.y_test)
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)
        expected_cols = [
            "Rank", "Model", "Holdout PR-AUC", "Holdout ROC-AUC",
            "Balanced Acc", "Precision", "Recall", "F1-Score",
            "Brier Score", "Log Loss"
        ]
        for col in expected_cols:
            self.assertIn(col, results.columns)

        # Leakage guard test
        X_leaked = self.X_test.copy()
        X_leaked["duration"] = 100
        with self.assertRaises(ValueError):
            evaluate_holdout_models(self.models_dict, X_leaked, self.y_test)

    def test_02_compute_threshold_curve(self):
        """Verify compute_threshold_curve returns trade-offs across cutoffs."""
        y_probs = self.logreg.predict_proba(self.X_test)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 9)
        curve = compute_threshold_curve(self.y_test, y_probs, thresholds=thresholds)

        self.assertIsInstance(curve, pd.DataFrame)
        self.assertEqual(len(curve), 9)
        for col in ["Threshold", "Precision", "Recall", "F1", "Balanced Acc", "Predicted Positive Rate"]:
            self.assertIn(col, curve.columns)

    def test_03_calculate_lift_and_gain(self):
        """Verify calculate_lift_and_gain constructs deciles and valid lift metrics."""
        y_probs = self.logreg.predict_proba(self.X_test)[:, 1]
        lift_df = calculate_lift_and_gain(self.y_test, y_probs, bins=10)

        self.assertIsInstance(lift_df, pd.DataFrame)
        self.assertEqual(len(lift_df), 10)
        self.assertEqual(lift_df["Cum Leads"].iloc[-1], self.n_samples)
        self.assertAlmostEqual(lift_df["Cum Contact %"].iloc[-1], 100.0, places=1)
        self.assertAlmostEqual(lift_df["Cum Capture %"].iloc[-1], 100.0, places=1)
        self.assertGreaterEqual(lift_df["Lift"].iloc[0], 0.0)

    def test_04_compute_calibration_metrics(self):
        """Verify compute_calibration_metrics computes reliability coords and ECE."""
        y_probs = self.logreg.predict_proba(self.X_test)[:, 1]
        calib = compute_calibration_metrics(self.y_test, y_probs, n_bins=5)

        self.assertIsInstance(calib, dict)
        self.assertIn("prob_true", calib)
        self.assertIn("prob_pred", calib)
        self.assertIn("brier_score", calib)
        self.assertIn("ece", calib)
        self.assertGreaterEqual(calib["ece"], 0.0)
        self.assertLessEqual(calib["ece"], 1.0)

    def test_05_evaluate_subgroup_performance(self):
        """Verify evaluate_subgroup_performance computes slices correctly."""
        y_probs = self.logreg.predict_proba(self.X_test)[:, 1]
        meta = pd.DataFrame({
            "age_group": np.random.choice(["<30", "30-50", ">50"], size=self.n_samples),
            "job": np.random.choice(["admin", "blue-collar", "technician"], size=self.n_samples),
        })

        res_age = evaluate_subgroup_performance(self.y_test, y_probs, meta, "age_group")
        self.assertIsInstance(res_age, pd.DataFrame)
        self.assertEqual(len(res_age), 3)
        for col in ["Subgroup", "Sample Count", "Prevalence", "FPR", "FNR", "Precision", "Recall"]:
            self.assertIn(col, res_age.columns)

        # Invalid column test
        with self.assertRaises(ValueError):
            evaluate_subgroup_performance(self.y_test, y_probs, meta, "nonexistent_col")


if __name__ == "__main__":
    unittest.main()
