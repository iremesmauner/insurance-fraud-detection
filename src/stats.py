"""Statistical helpers used in the exploratory analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


def cramers_v(table: pd.DataFrame, chi2: float) -> float:
    """Bias-corrected Cramér's V (Bergsma, 2013) for an r x c contingency table."""
    n = table.to_numpy().sum()
    r, c = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def chi_square_screen(df: pd.DataFrame, features: list[str], target: str,
                      alpha: float = 0.05) -> pd.DataFrame:
    """Chi-square test of independence between each feature and the target.

    P-values are adjusted with the Benjamini-Hochberg procedure because many
    hypotheses are tested at once. `min_expected` flags tables where the
    chi-square approximation is weak (expected count below 5).
    """
    rows = []
    for col in features:
        table = pd.crosstab(df[col], df[target])
        chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
        rows.append({
            "feature": col,
            "levels": table.shape[0],
            "chi2": chi2,
            "dof": dof,
            "p_value": p,
            "cramers_v": cramers_v(table, chi2),
            "min_expected": expected.min(),
        })
    out = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(out["p_value"], alpha=alpha, method="fdr_bh")
    out["p_adj_bh"] = p_adj
    out["significant"] = reject
    return out.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def rate_table(df: pd.DataFrame, feature: str, target: str,
               order: list | None = None) -> pd.DataFrame:
    """Fraud rate per level with a 95% Wilson confidence interval."""
    g = df.groupby(feature, observed=True)[target].agg(n="size", frauds="sum")
    if order is not None:
        g = g.reindex([lvl for lvl in order if lvl in g.index])
    g["rate"] = g["frauds"] / g["n"]
    low, high = proportion_confint(g["frauds"], g["n"], alpha=0.05, method="wilson")
    g["ci_low"], g["ci_high"] = low, high
    return g


def compare_numeric(df: pd.DataFrame, feature: str, target: str) -> dict:
    """Welch t-test and Mann-Whitney U for a numeric feature across the two classes."""
    a = df.loc[df[target] == 1, feature].dropna()
    b = df.loc[df[target] == 0, feature].dropna()
    t, p_t = stats.ttest_ind(a, b, equal_var=False)
    u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {
        "n_fraud": len(a), "n_legit": len(b),
        "mean_fraud": a.mean(), "mean_legit": b.mean(),
        "median_fraud": a.median(), "median_legit": b.median(),
        "welch_t": t, "p_welch": p_t,
        "mannwhitney_u": u, "p_mannwhitney": p_u,
        "cohens_d": (a.mean() - b.mean()) / pooled_sd,
        # Probability that a random fraud claim has a higher value than a legit one.
        "auc_effect": u / (len(a) * len(b)),
    }


def cochran_armitage_trend(table: pd.DataFrame, scores=None) -> tuple[float, float]:
    """Cochran-Armitage test for a linear trend in proportions.

    `table` has one row per ordered level and columns [0, 1] (legit, fraud).
    Returns the z statistic and the two-sided p-value.
    """
    n_i = table.sum(axis=1).to_numpy(dtype=float)
    x_i = table[1].to_numpy(dtype=float)
    s = np.arange(len(n_i), dtype=float) if scores is None else np.asarray(scores, float)
    n, p = n_i.sum(), x_i.sum() / n_i.sum()
    stat = np.sum(s * (x_i - n_i * p))
    var = p * (1 - p) * (np.sum(n_i * s**2) - np.sum(n_i * s) ** 2 / n)
    z = stat / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))
