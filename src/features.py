"""Feature encoders for the two model families."""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from . import config


class TreeEncoder(BaseEstimator, TransformerMixin):
    """Encode features for gradient-boosted trees and keep a readable frame.

    * binary columns   -> 0/1 (1 = the level named in config.BINARY_FEATURES)
    * ordinal columns  -> integer rank following the natural order
    * nominal columns  -> pandas ``category`` dtype (LightGBM handles these natively)

    Categories are learned on the training fold only; unseen levels become NaN,
    which LightGBM treats as missing.

    `extra_nominal` lists ordinal columns that should be treated as unordered
    instead (used for a sensitivity check in the explainability notebook).
    """

    def __init__(self, extra_nominal: tuple = ()):
        self.extra_nominal = extra_nominal

    def _is_nominal(self, col: str) -> bool:
        return col in config.NOMINAL_FEATURES or col in self.extra_nominal

    def fit(self, X: pd.DataFrame, y=None):
        self.nominal_categories_ = {
            col: sorted(X[col].dropna().unique().tolist(), key=str)
            for col in X.columns if self._is_nominal(col)
        }
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            if col in config.BINARY_FEATURES:
                out[col] = (X[col] == config.BINARY_FEATURES[col]).astype("int8")
            elif col in config.ORDINAL_FEATURES and not self._is_nominal(col):
                rank = {lvl: i for i, lvl in enumerate(config.ORDINAL_FEATURES[col])}
                out[col] = X[col].map(rank).astype("float64")
            else:
                out[col] = pd.Categorical(X[col], categories=self.nominal_categories_[col])
        return out

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names_in_)


def make_one_hot() -> OneHotEncoder:
    """One-hot encoder for logistic regression.

    Every feature is treated as categorical because the ordinal bands are not
    linear in the log-odds. Levels with fewer than 20 training rows are pooled
    into a single "infrequent" column to keep estimates stable.
    """
    return OneHotEncoder(
        min_frequency=20,
        handle_unknown="infrequent_if_exist",
        sparse_output=True,
    )
