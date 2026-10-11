"""
Unit and Contract Test Suite for BankFeatureEngineer Domain Transformations.
Validates sentinel mapping, bin boundary precision, type integrity, error handling,
and historical campaign cross-field inconsistency detection.
"""

import unittest
import numpy as np
import pandas as pd
from src.transformers import BankFeatureEngineer


class TestBankFeatureEngineerSentinelMapping(unittest.TestCase):
    """Validates that pdays == 999 and valid non-sentinel values map according to domain specification."""

    def setUp(self):
        self.transformer = BankFeatureEngineer()

    def test_all_pdays_999_map_to_not_contacted_and_previously_contacted_zero(self):
        """Validate all pdays == 999 records map to not_contacted and previously_contacted == 0."""
        df = pd.DataFrame({
            "pdays": [999, 999, 999, 999],
            "age": [25, 40, 55, 65]
        })
        out = self.transformer.transform(df)

        # pdays_group must be 'not_contacted' for all 999 records
        self.assertTrue((out["pdays_group"] == "not_contacted").all(),
                        "Expected all pdays == 999 records to map to 'not_contacted'.")
        # previously_contacted must be 0 for all 999 records
        self.assertTrue((out["previously_contacted"] == 0).all(),
                        "Expected all pdays == 999 records to have previously_contacted == 0.")
        # Raw pdays must be dropped
        self.assertNotIn("pdays", out.columns, "Raw 'pdays' column should be dropped.")

    def test_all_valid_contacted_records_map_to_previously_contacted_one(self):
        """Validate all other valid records (0 <= pdays <= 998) map to previously_contacted == 1."""
        valid_contacted_pdays = [0, 1, 4, 6, 7, 10, 14, 15, 27, 100, 500, 998]
        df = pd.DataFrame({
            "pdays": valid_contacted_pdays,
            "age": [35] * len(valid_contacted_pdays)
        })
        out = self.transformer.transform(df)

        # All non-999 records must have previously_contacted == 1
        self.assertTrue((out["previously_contacted"] == 1).all(),
                        "Expected all non-999 records to map to previously_contacted == 1.")
        # None of them should be categorized as 'not_contacted'
        self.assertFalse((out["pdays_group"] == "not_contacted").any(),
                         "Valid contacted records should never map to 'not_contacted'.")

    def test_mixed_dataset_sentinel_and_contacted_separation(self):
        """Validate exact separation on mixed sentinel and contacted records."""
        df = pd.DataFrame({
            "pdays": [999, 3, 999, 12, 999, 25, 998, 999],
            "age": [30] * 8
        })
        out = self.transformer.transform(df)

        expected_contacted = [0, 1, 0, 1, 0, 1, 1, 0]
        expected_groups = [
            "not_contacted", "0_to_6_days", "not_contacted",
            "7_to_14_days", "not_contacted", "15_plus_days",
            "15_plus_days", "not_contacted"
        ]
        self.assertEqual(list(out["previously_contacted"]), expected_contacted)
        self.assertEqual(list(out["pdays_group"]), expected_groups)


class TestBankFeatureEngineerBoundaryValues(unittest.TestCase):
    """Validates exact bin boundary mappings (0, 6, 7, 14, 15, 998, 999) and demographic age boundaries."""

    def setUp(self):
        self.transformer = BankFeatureEngineer()

    def test_boundary_values_pdays(self):
        """Validate boundary values such as 0, 6, 7, 14, 15, 998, and 999 map correctly."""
        boundary_df = pd.DataFrame({
            "pdays": [0, 6, 7, 14, 15, 998, 999],
            "age": [30] * 7
        })
        out = self.transformer.transform(boundary_df)

        expected_mapping = {
            0: "0_to_6_days",
            6: "0_to_6_days",
            7: "7_to_14_days",
            14: "7_to_14_days",
            15: "15_plus_days",
            998: "15_plus_days",
            999: "not_contacted"
        }
        for pdays_val, expected_grp in zip(boundary_df["pdays"], out["pdays_group"]):
            self.assertEqual(expected_grp, expected_mapping[pdays_val],
                             f"Boundary value pdays={pdays_val} mapped to {expected_grp}, expected {expected_mapping[pdays_val]}")

    def test_boundary_transitions_adjacent_integers(self):
        """Validate boundary transitions between adjacent integers (6 vs 7, 14 vs 15, 998 vs 999)."""
        df_transitions = pd.DataFrame({
            "pdays": [6, 7, 14, 15, 998, 999],
            "age": [30] * 6
        })
        out = self.transformer.transform(df_transitions)

        # 6 is in 0_to_6_days, 7 transitions to 7_to_14_days
        self.assertEqual(out["pdays_group"].iloc[0], "0_to_6_days")
        self.assertEqual(out["pdays_group"].iloc[1], "7_to_14_days")

        # 14 is in 7_to_14_days, 15 transitions to 15_plus_days
        self.assertEqual(out["pdays_group"].iloc[2], "7_to_14_days")
        self.assertEqual(out["pdays_group"].iloc[3], "15_plus_days")

        # 998 is in 15_plus_days, 999 transitions to not_contacted
        self.assertEqual(out["pdays_group"].iloc[4], "15_plus_days")
        self.assertEqual(out["pdays_group"].iloc[5], "not_contacted")

    def test_boundary_values_age(self):
        """Validate demographic age cohorts (<30, 30-39, 40-49, 50-59, 60+) at boundaries."""
        df_age = pd.DataFrame({
            "pdays": [999] * 8,
            "age": [18, 29, 30, 39, 40, 49, 50, 60]
        })
        out = self.transformer.transform(df_age)

        expected_age_groups = ["<30", "<30", "30-39", "30-39", "40-49", "40-49", "50-59", "60+"]
        self.assertEqual(list(out["age_group"]), expected_age_groups)

    def test_dual_age_representation_preserves_continuous_age(self):
        """Validate intentional dual representation: continuous 'age' is preserved alongside 'age_group'."""
        df = pd.DataFrame({
            "pdays": [999, 4, 10, 20],
            "age": [24, 34, 45, 65]
        })
        out = self.transformer.transform(df)
        self.assertIn("age", out.columns, "Continuous 'age' must be preserved in output under dual representation policy.")
        self.assertIn("age_group", out.columns, "Engineered 'age_group' must be present in output.")
        self.assertEqual(list(out["age"]), [24, 34, 45, 65])
        self.assertEqual(list(out["age_group"]), ["<30", "30-39", "40-49", "60+"])
        self.assertNotIn("pdays", out.columns, "Raw 'pdays' must be dropped.")


class TestBankFeatureEngineerNoNanStringAndCompleteness(unittest.TestCase):
    """Validates that no unassigned/NaN groups exist and no unexpected 'nan' strings are created."""

    def setUp(self):
        self.transformer = BankFeatureEngineer()

    def test_no_unassigned_or_nan_groups_in_output(self):
        """Validate there are no unassigned/NaN groups in transformed output."""
        df = pd.DataFrame({
            "pdays": [999, 0, 6, 7, 14, 15, 50, 998, 999],
            "age": [18, 29, 30, 39, 40, 49, 59, 60, 85]
        })
        out = self.transformer.transform(df)

        self.assertEqual(out["pdays_group"].isna().sum(), 0, "pdays_group should have 0 NaNs.")
        self.assertEqual(out["age_group"].isna().sum(), 0, "age_group should have 0 NaNs.")
        self.assertEqual(out["previously_contacted"].isna().sum(), 0, "previously_contacted should have 0 NaNs.")

    def test_no_unexpected_nan_string_values_created_by_astype_str(self):
        """Validate that no literal 'nan', 'NaN', 'None' strings can be produced by .astype(str)."""
        df = pd.DataFrame({
            "pdays": [999, 1, 6, 7, 14, 15, 998, 999],
            "age": [20, 35, 45, 55, 65, 75, 80, 25]
        })
        out = self.transformer.transform(df)

        forbidden_strings = {"nan", "NaN", "None", "null", "<NA>"}
        for col in ["pdays_group", "age_group"]:
            for forbidden in forbidden_strings:
                self.assertNotIn(forbidden, set(out[col]),
                                 f"Column '{col}' must not contain literal '{forbidden}' string value.")

    def test_only_authorized_domain_categories_produced(self):
        """Validate that engineered columns strictly draw from pre-defined authorized domain sets."""
        df = pd.DataFrame({
            "pdays": [0, 5, 6, 7, 12, 14, 15, 50, 998, 999],
            "age": [18, 25, 30, 35, 40, 45, 50, 55, 60, 70]
        })
        out = self.transformer.transform(df)

        authorized_pdays_groups = {"0_to_6_days", "7_to_14_days", "15_plus_days", "not_contacted"}
        authorized_age_groups = {"<30", "30-39", "40-49", "50-59", "60+"}

        self.assertTrue(set(out["pdays_group"]).issubset(authorized_pdays_groups),
                        f"Unexpected pdays_group categories: {set(out['pdays_group']) - authorized_pdays_groups}")
        self.assertTrue(set(out["age_group"]).issubset(authorized_age_groups),
                        f"Unexpected age_group categories: {set(out['age_group']) - authorized_age_groups}")


class TestBankFeatureEngineerInvalidInputHandling(unittest.TestCase):
    """Validates that BankFeatureEngineer raises or handles invalid input cleanly."""

    def setUp(self):
        self.transformer = BankFeatureEngineer()

    def test_negative_pdays_raises_value_error(self):
        """Validate negative pdays values raise clean ValueError."""
        df_neg = pd.DataFrame({"pdays": [-1, 5, 999], "age": [30, 40, 50]})
        with self.assertRaises(ValueError) as ctx:
            self.transformer.transform(df_neg)
        self.assertIn("negative values", str(ctx.exception).lower())

    def test_pdays_exceeding_999_raises_value_error(self):
        """Validate pdays > 999 raises clean ValueError."""
        df_large = pd.DataFrame({"pdays": [1000, 5, 999], "age": [30, 40, 50]})
        with self.assertRaises(ValueError) as ctx:
            self.transformer.transform(df_large)
        self.assertIn("exceeding sentinel 999", str(ctx.exception).lower())

    def test_pdays_with_nan_raises_value_error(self):
        """Validate pdays containing NaN raises clean ValueError."""
        df_nan = pd.DataFrame({"pdays": [np.nan, 5, 999], "age": [30, 40, 50]})
        with self.assertRaises(ValueError) as ctx:
            self.transformer.transform(df_nan)
        self.assertIn("nan/missing values", str(ctx.exception).lower())

    def test_pdays_non_numeric_raises_value_error(self):
        """Validate non-numeric pdays raises clean ValueError."""
        df_str = pd.DataFrame({"pdays": ["nine_nine_nine", "five"], "age": [30, 40]})
        with self.assertRaises(ValueError) as ctx:
            self.transformer.transform(df_str)
        self.assertIn("numeric", str(ctx.exception).lower())

    def test_missing_pdays_column_raises_value_error(self):
        """Validate missing required column 'pdays' raises clean ValueError."""
        df_missing = pd.DataFrame({"age": [30, 40, 50]})
        with self.assertRaises(ValueError) as ctx:
            self.transformer.transform(df_missing)
        self.assertIn("pdays", str(ctx.exception))

    def test_non_dataframe_input_raises_type_error(self):
        """Validate non-DataFrame input raises clean TypeError."""
        for invalid_input in [[999, 4], "invalid_string", None, 42]:
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(TypeError):
                    self.transformer.transform(invalid_input)

    def test_invalid_age_raises_value_error(self):
        """Validate invalid age (negative, > 120, NaN) raises clean ValueError."""
        # Age <= 0
        with self.assertRaises(ValueError):
            self.transformer.transform(pd.DataFrame({"pdays": [999], "age": [-5]}))
        with self.assertRaises(ValueError):
            self.transformer.transform(pd.DataFrame({"pdays": [999], "age": [0]}))
        # Age > 120
        with self.assertRaises(ValueError):
            self.transformer.transform(pd.DataFrame({"pdays": [999], "age": [125]}))
        # Age NaN
        with self.assertRaises(ValueError):
            self.transformer.transform(pd.DataFrame({"pdays": [999], "age": [np.nan]}))

    def test_empty_dataframe_handled_cleanly(self):
        """Validate empty DataFrame with expected schema returns empty DataFrame cleanly."""
        df_empty = pd.DataFrame({"pdays": pd.Series([], dtype=float), "age": pd.Series([], dtype=float)})
        out = self.transformer.transform(df_empty)
        self.assertEqual(len(out), 0)
        self.assertIn("previously_contacted", out.columns)
        self.assertIn("pdays_group", out.columns)
        self.assertIn("age_group", out.columns)
        self.assertNotIn("pdays", out.columns)


class TestBankFeatureEngineerCampaignInconsistencies(unittest.TestCase):
    """Validates detection and identification of contradictory historical campaign records."""

    def setUp(self):
        self.transformer = BankFeatureEngineer()

    def test_identify_inconsistent_records_legacy_crm_anomaly_4110(self):
        """Validate detection of previous > 0 BUT pdays == 999 (EDA-06 legacy CRM failure anomaly)."""
        df = pd.DataFrame({
            "pdays": [999, 999],
            "previous": [1, 3],
            "poutcome": ["failure", "failure"]
        })
        inconsistent_df = self.transformer.identify_inconsistent_records(df)
        self.assertEqual(len(inconsistent_df), 2, "Both records should be flagged as inconsistent.")
        self.assertTrue(
            inconsistent_df["inconsistency_reason"].str.contains("legacy CRM anomaly").all(),
            "Expected reason to cite legacy CRM anomaly."
        )

    def test_identify_inconsistent_records_untracked_contact(self):
        """Validate detection of pdays < 999 BUT previous == 0."""
        df = pd.DataFrame({
            "pdays": [5],
            "previous": [0],
            "poutcome": ["success"]
        })
        inconsistent_df = self.transformer.identify_inconsistent_records(df)
        self.assertEqual(len(inconsistent_df), 1)
        self.assertIn("pdays < 999 but previous == 0", inconsistent_df["inconsistency_reason"].iloc[0])

    def test_identify_inconsistent_records_nonexistent_outcome_with_contact(self):
        """Validate detection of pdays < 999 BUT poutcome == 'nonexistent'."""
        df = pd.DataFrame({
            "pdays": [6],
            "previous": [1],
            "poutcome": ["nonexistent"]
        })
        inconsistent_df = self.transformer.identify_inconsistent_records(df)
        self.assertEqual(len(inconsistent_df), 1)
        self.assertIn("poutcome == 'nonexistent'", inconsistent_df["inconsistency_reason"].iloc[0])

    def test_identify_inconsistent_records_success_with_sentinel_pdays(self):
        """Validate detection of poutcome == 'success' BUT pdays == 999."""
        df = pd.DataFrame({
            "pdays": [999],
            "previous": [0],
            "poutcome": ["success"]
        })
        inconsistent_df = self.transformer.identify_inconsistent_records(df)
        self.assertEqual(len(inconsistent_df), 1)
        self.assertTrue(
            "poutcome == 'success' but pdays == 999" in inconsistent_df["inconsistency_reason"].iloc[0] or
            "previous == 0 but poutcome != 'nonexistent'" in inconsistent_df["inconsistency_reason"].iloc[0]
        )

    def test_consistent_records_not_flagged(self):
        """Validate consistent virgin cold leads and warm leads are not flagged."""
        df_clean = pd.DataFrame({
            "pdays": [999, 5, 999, 14],
            "previous": [0, 1, 0, 2],
            "poutcome": ["nonexistent", "success", "nonexistent", "failure"]
        })
        inconsistent_df = self.transformer.identify_inconsistent_records(df_clean)
        self.assertEqual(len(inconsistent_df), 0, "Consistent records should produce zero flagged rows.")

        mask = self.transformer.identify_inconsistent_records(df_clean, return_mask=True)
        self.assertEqual(int(mask.sum()), 0)

    def test_flag_inconsistencies_pipeline_option(self):
        """Validate that flag_inconsistencies=True creates 'is_campaign_inconsistent' column."""
        bfe_flag = BankFeatureEngineer(flag_inconsistencies=True)
        df = pd.DataFrame({
            "pdays": [999, 999, 5],
            "previous": [0, 2, 1],
            "poutcome": ["nonexistent", "failure", "success"],
            "age": [30, 40, 50]
        })
        out = bfe_flag.transform(df)

        self.assertIn("is_campaign_inconsistent", out.columns)
        # Record 0: clean cold lead -> 0
        # Record 1: CRM anomaly (999 & previous=2) -> 1
        # Record 2: clean warm lead -> 0
        self.assertEqual(list(out["is_campaign_inconsistent"]), [0, 1, 0])
        self.assertEqual(bfe_flag.n_inconsistent_records_, 1)

    def test_identify_inconsistent_records_on_transformed_dataframe(self):
        """Validate that identify_inconsistent_records works even after raw pdays is dropped."""
        df = pd.DataFrame({
            "pdays": [999, 999, 5],
            "previous": [0, 2, 1],
            "poutcome": ["nonexistent", "failure", "success"],
            "age": [30, 40, 50]
        })
        transformed = self.transformer.transform(df)
        self.assertNotIn("pdays", transformed.columns)
        self.assertIn("previously_contacted", transformed.columns)

        # Check identification using previously_contacted
        mask = BankFeatureEngineer.identify_inconsistent_records(transformed, return_mask=True)
        self.assertEqual(list(mask), [False, True, False])


class TestBankFeatureEngineerDocumentation(unittest.TestCase):
    """Validates that domain behavior and invariants are thoroughly documented in BankFeatureEngineer."""

    def test_class_docstring_covers_domain_and_validation_rules(self):
        """Verify class docstring covers 999 mapping, bins, boundary behavior, and inconsistencies."""
        doc = BankFeatureEngineer.__doc__
        self.assertIsNotNone(doc)
        self.assertIn("previously_contacted", doc)
        self.assertIn("pdays_group", doc)
        self.assertIn("999", doc)
        self.assertIn("0_to_6_days", doc)
        self.assertIn("7_to_14_days", doc)
        self.assertIn("15_plus_days", doc)
        self.assertIn("not_contacted", doc)
        self.assertIn("nan", doc.lower())
        self.assertIn("inconsisten", doc.lower())


if __name__ == "__main__":
    unittest.main()
