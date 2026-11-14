"""Metrics, uncertainty estimates and the business cost model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def threshold_metrics(y_true, y_score, threshold: float) -> dict:
    """Confusion-matrix based metrics for a given decision threshold."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(y_score) >= threshold
    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    fn = int(np.sum(~flagged & (y_true == 1)))
    tn = int(np.sum(~flagged & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "flagged_share": (tp + fp) / len(y_true),
        "accuracy": (tp + tn) / len(y_true),
    }


def best_f1_threshold(y_true, y_score) -> float:
    """Threshold that maximises F1. Call it on out-of-fold scores, never on test data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def bootstrap_ci(y_true, y_score, metric=average_precision_score,
                 n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap confidence interval for a ranking metric."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    n, stats_ = len(y_true), []
    while len(stats_) < n_boot:
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():   # resample lost one class
            continue
        stats_.append(metric(y_true[idx], y_score[idx]))
    low, high = np.quantile(stats_, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, y_score)), float(low), float(high)


def paired_bootstrap_diff(y_true, score_a, score_b, metric=average_precision_score,
                          n_boot: int = 2000, seed: int = 42):
    """Bootstrap the difference metric(a) - metric(b) on the same resamples."""
    rng = np.random.default_rng(seed)
    y_true, score_a, score_b = map(np.asarray, (y_true, score_a, score_b))
    n, diffs = len(y_true), []
    while len(diffs) < n_boot:
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(metric(y_true[idx], score_a[idx]) - metric(y_true[idx], score_b[idx]))
    diffs = np.asarray(diffs)
    low, high = np.quantile(diffs, [0.025, 0.975])
    return {
        "diff": float(metric(y_true, score_a) - metric(y_true, score_b)),
        "ci_low": float(low), "ci_high": float(high),
        "share_a_better": float(np.mean(diffs > 0)),
    }


def ranking_summary(y_true, y_score) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "prevalence": float(np.mean(y_true)),
    }


def top_k_table(y_true, y_score, shares=(0.01, 0.02, 0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    """Precision, recall and lift when only the top-scored share of claims is reviewed."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    order = np.argsort(-y_score, kind="stable")
    base = y_true.mean()
    rows = []
    for share in shares:
        k = max(1, int(round(share * len(y_true))))
        hits = int(y_true[order[:k]].sum())
        rows.append({
            "reviewed_share": share, "claims_reviewed": k, "frauds_caught": hits,
            "precision": hits / k, "recall": hits / y_true.sum(),
            "lift": (hits / k) / base,
        })
    return pd.DataFrame(rows)


def cost_curve(y_true, y_score, loss_ratio: float, detection_rate: float = 1.0,
               thresholds=None) -> pd.DataFrame:
    """Expected cost per 1,000 claims for every threshold, in investigation-cost units.

    Cost model (one unit = the cost of investigating one claim):
      * every flagged claim is investigated            -> 1 unit
      * a flagged fraud is confirmed with probability `detection_rate`;
        otherwise it is paid out anyway                -> loss_ratio units
      * every fraud that is not flagged is paid out    -> loss_ratio units
    `loss_ratio` = average fraudulent payout / investigation cost.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if thresholds is None:
        thresholds = np.unique(np.concatenate([[0.0], np.sort(y_score), [1.0 + 1e-9]]))
    n, rows = len(y_true), []
    do_nothing = loss_ratio * y_true.sum()
    for thr in thresholds:
        m = threshold_metrics(y_true, y_score, thr)
        cost = (m["tp"] + m["fp"]) + loss_ratio * (m["fn"] + (1 - detection_rate) * m["tp"])
        rows.append({
            "threshold": thr, "cost_per_1000": 1000 * cost / n,
            "saving_per_1000": 1000 * (do_nothing - cost) / n,
            "precision": m["precision"], "recall": m["recall"],
            "flagged_share": m["flagged_share"],
        })
    return pd.DataFrame(rows)
