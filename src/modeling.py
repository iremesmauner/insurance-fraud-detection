"""Model pipelines and cross-validation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTEN
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config
from .features import TreeEncoder, make_one_hot

# Class-imbalance strategies compared in the modelling notebook.
STRATEGIES = ["none", "class_weight", "undersample", "smoten"]


def make_cv(n_splits: int = config.N_SPLITS) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)


def _sampler(strategy: str):
    """Resampler for a strategy, or None when the strategy does not resample."""
    if strategy == "undersample":
        return RandomUnderSampler(random_state=config.RANDOM_STATE)
    if strategy == "smoten":
        # SMOTE-N generates synthetic minority rows for purely nominal data.
        # Plain SMOTE would interpolate between category codes, which is meaningless here.
        return SMOTEN(random_state=config.RANDOM_STATE)
    return None


def make_logreg(strategy: str = "class_weight", C: float = 1.0) -> Pipeline:
    """Regularised logistic regression on one-hot encoded features."""
    steps = []
    sampler = _sampler(strategy)
    if sampler is not None:
        steps.append(("resample", sampler))
    steps.append(("encode", make_one_hot()))
    steps.append(("model", LogisticRegression(
        C=C,
        class_weight="balanced" if strategy == "class_weight" else None,
        max_iter=2000,
        random_state=config.RANDOM_STATE,
    )))
    return Pipeline(steps)


DEFAULT_LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
)


def make_lgbm(strategy: str = "class_weight", **params) -> Pipeline:
    """LightGBM classifier on tree-friendly encoded features."""
    lgbm_params = {**DEFAULT_LGBM_PARAMS, **params}
    steps = []
    sampler = _sampler(strategy)
    if sampler is not None:
        steps.append(("resample", sampler))
    steps.append(("encode", TreeEncoder()))
    steps.append(("model", LGBMClassifier(
        **lgbm_params,
        class_weight="balanced" if strategy == "class_weight" else None,
        random_state=config.RANDOM_STATE,
        n_jobs=1,          # single thread keeps results reproducible
        verbose=-1,
    )))
    return Pipeline(steps)


def cross_validate(pipeline, X: pd.DataFrame, y: pd.Series, cv=None):
    """Stratified CV returning per-fold scores and out-of-fold probabilities.

    Any resampling step lives inside the pipeline, so it is fitted on the
    training folds only and the validation fold keeps the real class ratio.
    """
    cv = cv or make_cv()
    oof = np.zeros(len(y), dtype=float)
    folds = []
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        model = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        folds.append({
            "fold": fold,
            "pr_auc": average_precision_score(y.iloc[va], oof[va]),
            "roc_auc": roc_auc_score(y.iloc[va], oof[va]),
        })
    return pd.DataFrame(folds), oof
