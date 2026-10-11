"""
Automated Unit and Contract Tests for Modeling Utilities (src/modeling.py).
Conforms to specs/006-model-development/contracts/modeling_contract.md.
"""

import os
import shutil
import unittest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.modeling import (
    evaluate_cv_model,
    compile_cv_benchmark_table,
    export_candidate_bundle,
)


class TestModelingContract(unittest.TestCase):
    """Test suite validating behavior of src/modeling.py functions."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n_samples = 200
        cls.n_features = 10
        cls.X = pd.DataFrame(
            np.random.randn(cls.n_samples, cls.n_features),
            columns=[f"feat_{i}" for i in range(cls.n_features)]
        )
        cls.y = pd.Series(
            np.random.binomial(1, 0.15, size=cls.n_samples),
            name="y"
        )
        cls.test_dir = os.path.join(PROJECT_ROOT, "tests", "tmp_models")
        os.makedirs(cls.test_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_01_evaluate_cv_model_returns_correct_structure(self):
        """Verify evaluate_cv_model returns fold_df and summary dict with all expected metrics."""
        clf = DummyClassifier(strategy="prior")
        fold_df, summary = evaluate_cv_model(
            estimator=clf,
            X=self.X,
            y=self.y,
            model_name="Dummy Baseline",
            family="Baseline",
            random_state=42
        )

        # Check fold_df shape and columns
        self.assertIsInstance(fold_df, pd.DataFrame)
        self.assertEqual(len(fold_df), 5, "Should have 5 folds")

        expected_cols = [
            "model_name", "family", "fold",
            "train_pr_auc", "val_pr_auc",
            "train_roc_auc", "val_roc_auc",
            "train_balanced_acc", "val_balanced_acc",
            "train_f1_macro", "val_f1_macro",
            "train_f1_minority", "val_f1_minority",
            "train_brier_score", "val_brier_score",
            "fit_time_sec", "score_time_sec"
        ]
        for col in expected_cols:
            self.assertIn(col, fold_df.columns, f"Missing column {col} in fold_df")

        # Check summary dict
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["model_name"], "Dummy Baseline")
        self.assertIn("val_pr_auc_mean", summary)
        self.assertIn("val_pr_auc_std", summary)
        self.assertIn("train_val_pr_auc_gap", summary)
        self.assertAlmostEqual(summary["train_val_pr_auc_gap"], summary["train_pr_auc_mean"] - summary["val_pr_auc_mean"], places=5)

    def test_02_compile_cv_benchmark_table(self):
        """Verify compile_cv_benchmark_table correctly aggregates and ranks candidate models."""
        dummy = DummyClassifier(strategy="prior")
        logreg = LogisticRegression(random_state=42, max_iter=200)

        fold_df_dummy, _ = evaluate_cv_model(dummy, self.X, self.y, "Dummy", "Baseline")
        fold_df_logreg, _ = evaluate_cv_model(logreg, self.X, self.y, "LogReg", "Linear")

        table = compile_cv_benchmark_table([fold_df_dummy, fold_df_logreg])

        self.assertIsInstance(table, pd.DataFrame)
        self.assertEqual(len(table), 2)
        self.assertIn("Rank", table.columns)
        self.assertIn("Model", table.columns)
        self.assertIn("Val PR-AUC (Mean ± Std)", table.columns)
        self.assertIn("Val ROC-AUC (Mean ± Std)", table.columns)
        self.assertEqual(table["Rank"].iloc[0], 1)

    def test_03_export_candidate_bundle(self):
        """Verify export_candidate_bundle persists model bundles and tables to disk."""
        logreg = LogisticRegression(random_state=42, max_iter=200).fit(self.X, self.y)
        models_dict = {"Logistic Regression": logreg}
        dummy_table = pd.DataFrame([{"Rank": 1, "Model": "Logistic Regression", "Val PR-AUC": 0.45}])

        saved = export_candidate_bundle(models_dict, dummy_table, export_dir=self.test_dir)

        self.assertIn("bundle_path", saved)
        self.assertIn("csv_path", saved)
        self.assertTrue(os.path.exists(saved["bundle_path"]), f"Missing file {saved['bundle_path']}")
        self.assertTrue(os.path.exists(saved["csv_path"]), f"Missing file {saved['csv_path']}")


if __name__ == "__main__":
    unittest.main()
