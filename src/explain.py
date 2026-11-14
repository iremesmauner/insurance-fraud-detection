"""SHAP helpers: compute values for the LightGBM pipeline and draw readable explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import plotting


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def shap_frame(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """SHAP values (log-odds scale) for every row of X, plus the model's base value.

    TreeExplainer is exact for tree ensembles. Values are additive:
    base_value + sum(row) equals the model's raw log-odds output for that row.
    """
    encoder, model = pipeline.named_steps["encode"], pipeline.named_steps["model"]
    encoded = encoder.transform(X)
    explanation = shap.TreeExplainer(model)(encoded)
    values = explanation.values
    if values.ndim == 3:            # some versions return one slice per class
        values = values[:, :, 1]
    base = float(np.ravel(explanation.base_values)[0])
    return pd.DataFrame(values, index=X.index, columns=encoded.columns), base


def local_table(shap_row: pd.Series, feature_row: pd.Series, top: int = 8) -> pd.DataFrame:
    """Largest contributions for one claim, with the remaining features pooled."""
    order = shap_row.abs().sort_values(ascending=False).index
    head, tail = order[:top], order[top:]
    rows = [{"label": f"{name} = {feature_row[name]}", "shap": shap_row[name]} for name in head]
    rows.append({"label": f"{len(tail)} other features", "shap": shap_row[tail].sum()})
    return pd.DataFrame(rows)


def plot_local(ax, shap_row: pd.Series, feature_row: pd.Series, base: float,
               title: str, top: int = 8) -> None:
    """Horizontal bar chart of one claim's contributions (orange raises risk, blue lowers it)."""
    table = local_table(shap_row, feature_row, top).iloc[::-1]
    colors = [plotting.ORANGE if v > 0 else plotting.BLUE for v in table["shap"]]
    ax.barh(table["label"], table["shap"], color=colors, height=0.6)
    ax.axvline(0, color=plotting.INK_MUTED, linewidth=1)
    span = max(table["shap"].abs().max(), 0.1)
    for y, value in enumerate(table["shap"]):
        ax.text(value + (0.03 * span if value >= 0 else -0.03 * span), y, f"{value:+.2f}",
                va="center", ha="left" if value >= 0 else "right", color=plotting.INK, fontsize=9)
    ax.set_xlim(-1.45 * span, 1.45 * span)
    probability = sigmoid(base + shap_row.sum())
    ax.set_title(f"{title}\nPredicted fraud probability: {probability:.1%}", fontsize=11)
    ax.set_xlabel("SHAP value (change in log-odds)")
    ax.grid(axis="y", visible=False)


def level_effects(shap_values: pd.DataFrame, X: pd.DataFrame, feature: str,
                  order: list | None = None) -> pd.DataFrame:
    """Mean SHAP value and row count per level of a categorical feature."""
    out = (pd.DataFrame({"level": X[feature].to_numpy(), "shap": shap_values[feature].to_numpy()})
           .groupby("level")["shap"].agg(mean_shap="mean", n="size"))
    if order is not None:
        out = out.reindex([lvl for lvl in order if lvl in out.index])
    return out
