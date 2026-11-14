"""Loading, cleaning and splitting the raw claims data."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path=config.RAW_DATA) -> pd.DataFrame:
    """Read the raw CSV exactly as distributed (the file starts with a UTF-8 BOM)."""
    return pd.read_csv(path, encoding="utf-8-sig")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented cleaning rules and return a new frame.

    Rules
    -----
    1. Drop the single record whose claim date is coded "0" (no claim day/month).
    2. Recode Age == 0 as missing; these 320 rows all belong to the youngest age band.
    3. Add ClaimLagMonths: months between accident and claim, capped at 3.
    """
    df = raw.copy()

    invalid_claim_date = (df["DayOfWeekClaimed"] == "0") | (df["MonthClaimed"] == "0")
    df = df.loc[~invalid_claim_date].reset_index(drop=True)

    df["Age"] = df["Age"].astype("float64").where(df["Age"] > 0)

    month_index = {m: i for i, m in enumerate(config.MONTHS)}
    lag = (df["MonthClaimed"].map(month_index) - df["Month"].map(month_index)) % 12
    df["ClaimLagMonths"] = lag.clip(upper=3).astype("int64")

    return df


def load_clean() -> pd.DataFrame:
    """Convenience wrapper: raw file -> cleaned frame."""
    return clean(load_raw())


def split_features_target(df: pd.DataFrame):
    """Return the model feature frame and the binary target."""
    return df[config.FEATURES].copy(), df[config.TARGET].astype(int).copy()


def train_test(df: pd.DataFrame):
    """Stratified hold-out split shared by every notebook.

    Returns X_train, X_test, y_train, y_test with the original row index preserved,
    so that predictions can be joined back to the cleaned frame.
    """
    X, y = split_features_target(df)
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
