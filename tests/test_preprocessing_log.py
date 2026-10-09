"""
Automated Contract & Concordance Validation Test Suite for Preprocessing Decision Log.
Conforms to specs/003-document-preprocessing-decisions/contracts/preprocessing-log-contract.md.
"""

import os
import re
import unittest
import pandas as pd
from src.transformers import BankFeatureEngineer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOC_PATH = os.path.join(PROJECT_ROOT, "docs", "preprocessing_feature_log.md")

class TestPreprocessingDecisionLogContract(unittest.TestCase):
    """Validates structure, schema, completeness, and code concordance of preprocessing log."""

    def test_01_document_exists_and_non_empty(self):
        """Verify docs/preprocessing_feature_log.md exists and is substantive."""
        self.assertTrue(os.path.exists(DOC_PATH), f"Missing document at {DOC_PATH}")
        with open(DOC_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertGreater(len(content), 1500, "Document content is too short to be a valid evidence log.")

    def test_02_all_ten_decisions_present(self):
        """Verify all 10 core decision IDs (PRE-01 to PRE-10) are present in the document."""
        with open(DOC_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        for i in range(1, 11):
            pre_id = f"PRE-{i:02d}"
            self.assertIn(pre_id, content, f"Missing decision identifier {pre_id} in preprocessing log.")

    def test_03_master_table_schema(self):
        """Verify master table contains required 7 columns: ID, Decision, Alternatives, Selected, Reason, Evidence, Status."""
        with open(DOC_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        table_header = None
        for line in lines:
            if "| ID |" in line and "| Decision |" in line and "| Alternatives |" in line:
                table_header = line
                break

        self.assertIsNotNone(table_header, "Master table header not found in docs/preprocessing_feature_log.md")
        cols = [c.strip() for c in table_header.strip().strip("|").split("|")]
        self.assertEqual(len(cols), 7, f"Expected 7 columns, found {len(cols)}: {cols}")
        expected_cols = ["ID", "Decision", "Alternatives", "Selected", "Reason", "Evidence", "Status"]
        for exp in expected_cols:
            self.assertTrue(any(exp.lower() in c.lower() for c in cols), f"Expected column '{exp}' in table header.")

    def test_04_alternatives_and_statuses(self):
        """Verify each row has at least two alternatives and a valid status code."""
        with open(DOC_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        valid_statuses = ["ratified", "revised", "tunable", "under investigation"]
        decision_rows = {}
        for line in lines:
            match = re.search(r"\|\s*\*{0,2}(PRE-\d{2})\*{0,2}\s*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|", line)
            if match:
                pre_id = match.group(1).strip()
                decision = match.group(2).strip()
                alternatives = match.group(3).strip()
                selected = match.group(4).strip()
                reason = match.group(5).strip()
                evidence = match.group(6).strip()
                status = match.group(7).strip()
                decision_rows[pre_id] = {
                    "decision": decision,
                    "alternatives": alternatives,
                    "selected": selected,
                    "reason": reason,
                    "evidence": evidence,
                    "status": status,
                }

        self.assertEqual(len(decision_rows), 10, f"Expected 10 decision rows parsed, got {len(decision_rows)}")
        for pre_id, data in decision_rows.items():
            # Check alternatives multiplicity
            alt_text = data["alternatives"]
            self.assertTrue("/" in alt_text or ";" in alt_text or "vs" in alt_text.lower(),
                            f"{pre_id} alternatives column should list multiple alternatives: '{alt_text}'")
            # Check status
            status_text = data["status"].lower()
            self.assertTrue(any(v in status_text for v in valid_statuses),
                            f"{pre_id} has invalid status '{data['status']}'")

    def test_05_cross_reference_files_exist(self):
        """Verify related documentation and code assets exist on disk."""
        eda_log = os.path.join(PROJECT_ROOT, "docs", "eda_insight_log.md")
        dec_log = os.path.join(PROJECT_ROOT, "docs", "decision_log.md")
        trans_file = os.path.join(PROJECT_ROOT, "src", "transformers.py")
        self.assertTrue(os.path.exists(eda_log), "docs/eda_insight_log.md missing")
        self.assertTrue(os.path.exists(dec_log), "docs/decision_log.md missing")
        self.assertTrue(os.path.exists(trans_file), "src/transformers.py missing")

    def test_06_transformer_code_concordance(self):
        """Verify BankFeatureEngineer implements transformations declared in PRE-02, 07, 08, 09, 10."""
        bfe = BankFeatureEngineer()
        df_sample = pd.DataFrame({
            "pdays": [999, 4, 10, 20],
            "age": [24, 34, 45, 65]
        })
        out = bfe.transform(df_sample)

        # PRE-08: previously_contacted created
        self.assertIn("previously_contacted", out.columns)
        self.assertEqual(list(out["previously_contacted"]), [0, 1, 1, 1])

        # PRE-07: pdays_group created
        self.assertIn("pdays_group", out.columns)
        self.assertEqual(list(out["pdays_group"]), ["not_contacted", "0_to_6_days", "7_to_14_days", "15_plus_days"])

        # PRE-09: age_group created
        self.assertIn("age_group", out.columns)
        self.assertEqual(list(out["age_group"]), ["<30", "30-39", "40-49", "60+"])

        # PRE-10: raw pdays dropped
        self.assertNotIn("pdays", out.columns)

if __name__ == "__main__":
    unittest.main()
