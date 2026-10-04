"""
Custom Scikit-Learn transformers for Bank Marketing response prediction.
Maintained under Phase 2: Data Engineering (Athukorala M. B. / IT24101937).
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for Bank Marketing domain features:
    1. Derives binary flag 'previously_contacted' from pdays (1 if pdays != 999 else 0).
    2. Categorically bins pdays into 'pdays_group' and drops raw pdays.
    3. Categorically bins age into demographic life-stage cohorts 'age_group'.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # 1. Previously contacted indicator
        X_out['previously_contacted'] = (X_out['pdays'] != 999).astype(int)

        # 2. pdays categorical recency bins
        pdays_bins = [-1, 6, 14, 998, 1000]
        pdays_labels = ['0_to_6_days', '7_to_14_days', '15_plus_days', 'not_contacted']
        X_out['pdays_group'] = pd.cut(
            X_out['pdays'],
            bins=pdays_bins,
            labels=pdays_labels,
            right=True
        ).astype(str)

        # 3. Demographic life-stage cohorts
        age_bins = [0, 29, 39, 49, 59, 120]
        age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
        X_out['age_group'] = pd.cut(
            X_out['age'],
            bins=age_bins,
            labels=age_labels,
            right=True
        ).astype(str)

        # Discard raw pdays column to prevent sentinel 999 from distorting scale
        X_out = X_out.drop(columns=['pdays'])

        return X_out
