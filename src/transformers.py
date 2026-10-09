"""
Custom Scikit-Learn transformers for Bank Marketing response prediction.
Maintained under Phase 2: Data Engineering (Athukorala M. B. / IT24101937).
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for Bank Marketing domain features.

    Implements verified domain transformations and invariant integrity checks
    conforming to PRE-02, PRE-07, PRE-08, PRE-09, and PRE-10:
    
    1. 'previously_contacted' (binary indicator):
       - Explicitly maps sentinel pdays == 999 records to 0 (never contacted in prior campaign).
       - Explicitly maps all valid elapsed pdays records (0 <= pdays <= 998) to 1.
    
    2. 'pdays_group' (discrete recency cohorts):
       - Bins valid elapsed pdays and sentinel 999 into non-overlapping semantic intervals:
         * 0 to 6 days inclusive   -> '0_to_6_days'
         * 7 to 14 days inclusive  -> '7_to_14_days'
         * 15 to 998 days inclusive -> '15_plus_days'
         * 999 days (sentinel)     -> 'not_contacted'
       - Enforces strict boundary validation ensuring no unassigned/NaN groups exist.
       - Guarantees that no unexpected 'nan' string values are produced by string conversion.
       - Drops the raw numeric 'pdays' column to prevent sentinel 999 scale distortion.
    
    3. 'age_group' (demographic life-stage cohorts & intentional dual representation):
       - If 'age' is present in the input DataFrame, categorizes continuous age into:
         * age <= 29               -> '<30'
         * 30 <= age <= 39         -> '30-39'
         * 40 <= age <= 49         -> '40-49'
         * 50 <= age <= 59         -> '50-59'
         * 60 <= age <= 120        -> '60+'
       - Intentional Dual Representation (PRE-09, ADR-004):
         * Retains both continuous 'age' (scaled via StandardScaler) and categorical 'age_group'
           (one-hot encoded via OneHotEncoder) in the pipeline output.
         * Categorical 'age_group' equips linear models (Logistic Regression) to fit independent
           step-function coefficients capturing the empirical U-shaped conversion curve (youth <25:
           23.97%, seniors 60+: 39.56% vs 40-49: 7.92%, EDA-07).
         * Continuous 'age' is deliberately preserved (unlike 'pdays', which is dropped below to
           eliminate sentinel 999 distortion) to provide granular split points for tree ensembles
           (Random Forest, HistGradientBoosting) and localized within-bracket ranking.
         * Collinearity between continuous age and indicator bins is managed via regularized estimation
           (L2 Ridge / ElasticNet), which is mandatory under project modeling standards.
         * Retention Invariant: Neither representation shall be removed from the pipeline until
           empirical multi-model cross-validation ablation evidence justifies pruning.

    4. Campaign Cross-Field Inconsistency Identification:
       - Identifies anomalous records across historical campaign columns ('pdays', 'previous', 'poutcome').
       - Detects the legacy CRM failure anomaly (EDA-06: previous > 0 BUT pdays == 999, N=4,110),
         untracked prior outreach (pdays < 999 BUT previous == 0), and outcome mismatches.
       - Accessible via `identify_inconsistent_records(X)` or instance tracking.

    Parameters
    ----------
    validate_input : bool, default=True
        Whether to enforce strict input validation on ranges, types, and missingness.
    flag_inconsistencies : bool, default=False
        If True, appends a binary 'is_campaign_inconsistent' column during transform().
        Defaults to False to maintain backward compatibility with standard ColumnTransformer schemas.
    """

    # Domain bin edges and category labels
    PDAYS_BINS = [-1, 6, 14, 998, 999]
    PDAYS_LABELS = ['0_to_6_days', '7_to_14_days', '15_plus_days', 'not_contacted']

    AGE_BINS = [0, 29, 39, 49, 59, 120]
    AGE_LABELS = ['<30', '30-39', '40-49', '50-59', '60+']

    def __init__(self, validate_input: bool = True, flag_inconsistencies: bool = False):
        self.validate_input = validate_input
        self.flag_inconsistencies = flag_inconsistencies
        self.inconsistent_indices_ = None
        self.n_inconsistent_records_ = 0

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit method (stateless transformer).
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : None
            Ignored.
            
        Returns
        -------
        self : BankFeatureEngineer
        """
        if self.validate_input:
            self._validate_schema(X)
        return self

    def _validate_schema(self, X: pd.DataFrame) -> None:
        """Validates that input X is a DataFrame and contains required columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input X must be a pandas DataFrame, got {type(X).__name__}")
        if 'pdays' not in X.columns:
            raise ValueError("Input DataFrame is missing required column: 'pdays'")

    def _validate_data_values(self, X: pd.DataFrame) -> None:
        """
        Validates column types, missingness, and domain ranges to prevent silent corruption.
        Raises ValueError cleanly on invalid input.
        """
        # 1. pdays validation
        pdays = X['pdays']
        if not pd.api.types.is_numeric_dtype(pdays):
            raise ValueError(f"Column 'pdays' must be numeric, got dtype {pdays.dtype}")
        
        if pdays.isna().any():
            raise ValueError("Column 'pdays' contains NaN/missing values. Unassigned/NaN groups are not allowed.")

        if (pdays < 0).any():
            invalid_vals = pdays[pdays < 0].unique()
            raise ValueError(f"Column 'pdays' contains negative values: {invalid_vals}. Valid domain range is [0, 999].")

        if (pdays > 999).any():
            invalid_vals = pdays[pdays > 999].unique()
            raise ValueError(f"Column 'pdays' contains values exceeding sentinel 999: {invalid_vals}. Valid domain range is [0, 999].")

        # 2. age validation (if present)
        if 'age' in X.columns:
            age = X['age']
            if not pd.api.types.is_numeric_dtype(age):
                raise ValueError(f"Column 'age' must be numeric, got dtype {age.dtype}")
            if age.isna().any():
                raise ValueError("Column 'age' contains NaN/missing values. Unassigned/NaN groups are not allowed.")
            if (age <= 0).any() or (age > 120).any():
                invalid_ages = age[(age <= 0) | (age > 120)].unique()
                raise ValueError(f"Column 'age' contains invalid values outside (0, 120]: {invalid_ages}")

    @classmethod
    def identify_inconsistent_records(
        cls, X: pd.DataFrame, return_mask: bool = False
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Identifies records with contradictory historical campaign metadata.
        
        Evaluates logical invariants across:
          - 'pdays' (or transformed 'previously_contacted')
          - 'previous'
          - 'poutcome'

        Recognized inconsistencies:
          1. previous > 0 BUT pdays == 999 (legacy CRM anomaly where failed prior calls coded 999, EDA-06)
          2. pdays < 999 BUT previous == 0 (recorded elapsed days without recorded contact count)
          3. pdays < 999 BUT poutcome == 'nonexistent' (recent contact logged with nonexistent outcome)
          4. poutcome == 'success' BUT pdays == 999 (successful prior campaign logged with sentinel days)
          5. previous == 0 BUT poutcome != 'nonexistent' (zero contact count with non-trivial outcome)
          6. previous > 0 BUT poutcome == 'nonexistent' (positive contact count with nonexistent outcome)

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing some or all historical campaign columns.
        return_mask : bool, default=False
            If True, returns a boolean pd.Series indicating rows with inconsistencies.
            If False, returns a filtered DataFrame of inconsistent rows with an 'inconsistency_reason' column.

        Returns
        -------
        Union[pd.DataFrame, pd.Series]
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input X must be a pandas DataFrame, got {type(X).__name__}")

        has_pdays = 'pdays' in X.columns
        has_prev_contacted = 'previously_contacted' in X.columns
        has_previous = 'previous' in X.columns
        has_poutcome = 'poutcome' in X.columns

        if not (has_pdays or has_prev_contacted) and not (has_previous or has_poutcome):
            raise ValueError(
                "Input DataFrame lacks historical campaign columns ('pdays', 'previously_contacted', 'previous', 'poutcome')."
            )

        # Determine contact status
        if has_pdays:
            is_sentinel_pdays = (X['pdays'] == 999)
            is_recent_pdays = (X['pdays'] != 999)
        elif has_prev_contacted:
            is_sentinel_pdays = (X['previously_contacted'] == 0)
            is_recent_pdays = (X['previously_contacted'] == 1)
        else:
            is_sentinel_pdays = None
            is_recent_pdays = None

        mask = pd.Series(False, index=X.index)
        reasons = pd.Series("", index=X.index, dtype=object)

        def _add_condition(cond, reason_label):
            nonlocal mask, reasons
            if cond is not None:
                matched = cond.fillna(False)
                mask |= matched
                reasons = reasons.where(
                    ~matched,
                    reasons.apply(lambda r: f"{r}; {reason_label}" if r else reason_label)
                )

        if is_sentinel_pdays is not None and has_previous:
            _add_condition((X['previous'] > 0) & is_sentinel_pdays, "previous > 0 but pdays == 999 (legacy CRM anomaly)")
            _add_condition((X['previous'] == 0) & is_recent_pdays, "pdays < 999 but previous == 0")

        if is_recent_pdays is not None and has_poutcome:
            _add_condition(is_recent_pdays & (X['poutcome'] == 'nonexistent'), "pdays < 999 but poutcome == 'nonexistent'")

        if is_sentinel_pdays is not None and has_poutcome:
            _add_condition(is_sentinel_pdays & (X['poutcome'] == 'success'), "poutcome == 'success' but pdays == 999")

        if has_previous and has_poutcome:
            _add_condition((X['previous'] == 0) & (X['poutcome'] != 'nonexistent'), "previous == 0 but poutcome != 'nonexistent'")
            _add_condition((X['previous'] > 0) & (X['poutcome'] == 'nonexistent'), "previous > 0 but poutcome == 'nonexistent'")

        if return_mask:
            return mask

        out = X[mask].copy()
        out['inconsistency_reason'] = reasons[mask]
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input DataFrame by engineering domain features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features containing 'pdays' and optionally 'age', 'previous', 'poutcome'.
            
        Returns
        -------
        X_out : pd.DataFrame
            Transformed features with 'previously_contacted', 'pdays_group', 'age_group'
            and with raw 'pdays' removed.
        """
        self._validate_schema(X)
        if self.validate_input:
            self._validate_data_values(X)

        X_out = X.copy()

        # Track campaign inconsistencies if previous/poutcome are available
        can_check_inconsistencies = any(c in X_out.columns for c in ['previous', 'poutcome'])
        if can_check_inconsistencies:
            inconsistent_mask = self.identify_inconsistent_records(X_out, return_mask=True)
            self.inconsistent_indices_ = X_out.index[inconsistent_mask].tolist()
            self.n_inconsistent_records_ = int(inconsistent_mask.sum())
            if self.flag_inconsistencies:
                X_out['is_campaign_inconsistent'] = inconsistent_mask.astype(int)
        elif self.flag_inconsistencies:
            X_out['is_campaign_inconsistent'] = 0

        # 1. Previously contacted indicator:
        # All pdays == 999 map to 0; all other valid records (0 <= pdays <= 998) map to 1.
        X_out['previously_contacted'] = (X_out['pdays'] != 999).astype(int)

        # 2. pdays categorical recency bins:
        # Discretize valid elapsed days into semantic cohorts and sentinel into 'not_contacted'.
        pdays_cut = pd.cut(
            X_out['pdays'],
            bins=self.PDAYS_BINS,
            labels=self.PDAYS_LABELS,
            right=True
        )

        # Verify no unassigned / NaN groups were produced
        if pdays_cut.isna().any():
            raise ValueError("Unassigned bins (NaN) detected in pdays_group. Valid pdays range is [0, 999].")

        # Safely convert to string and verify no literal 'nan' string values were created
        pdays_str = pdays_cut.astype(str)
        if (pdays_str == 'nan').any():
            raise ValueError("Unexpected 'nan' string value produced by .astype(str) in pdays_group.")
        X_out['pdays_group'] = pdays_str

        # 3. Demographic life-stage cohorts & intentional dual representation (if 'age' present):
        # - Dual Representation Rationale (PRE-09, ADR-004):
        #   Discretizes continuous age into semantic life-stage cohorts ('age_group') to allow
        #   linear models (e.g. Logistic Regression) to fit step-function lifts capturing the
        #   empirical U-shaped response profile (youth <25: 23.97%, seniors 60+: 39.56% vs
        #   middle-aged 40-49: 7.92%, EDA-07).
        # - Retention Invariant:
        #   Raw continuous 'age' is DELIBERATELY PRESERVED in X_out (unlike 'pdays', which is
        #   dropped below to eliminate sentinel 999 scale distortion). Preserving continuous 'age'
        #   provides granular metric distances and within-bracket split resolution for tree-based
        #   ensembles (Random Forest, HistGradientBoosting) and continuous StandardScaler scaling.
        # - Neither representation shall be removed until supported by empirical model ablation evidence.
        if 'age' in X_out.columns:
            age_cut = pd.cut(
                X_out['age'],
                bins=self.AGE_BINS,
                labels=self.AGE_LABELS,
                right=True
            )
            if age_cut.isna().any():
                raise ValueError("Unassigned bins (NaN) detected in age_group. Valid age range is (0, 120].")

            age_str = age_cut.astype(str)
            if (age_str == 'nan').any():
                raise ValueError("Unexpected 'nan' string value produced by .astype(str) in age_group.")
            X_out['age_group'] = age_str

        # 4. Discard raw pdays column to prevent sentinel 999 from distorting scale
        X_out = X_out.drop(columns=['pdays'])

        return X_out
