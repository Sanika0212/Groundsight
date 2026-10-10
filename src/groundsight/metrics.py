"""Small, dependency-free classification metrics and intervals."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def auroc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the ROC curve: P(score of a positive > score of a negative), ties = 1/2."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confusion(pred: ArrayLike, labels: ArrayLike) -> tuple[int, int, int, int]:
    """``(tp, fp, tn, fn)``."""
    p = np.asarray(pred, dtype=bool)
    y = np.asarray(labels, dtype=bool)
    return int((p & y).sum()), int((p & ~y).sum()), int((~p & ~y).sum()), int((~p & y).sum())


def balanced_accuracy(pred: ArrayLike, labels: ArrayLike) -> float:
    tp, fp, tn, fn = confusion(pred, labels)
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return (recall + specificity) / 2


def precision_recall_f1(pred: ArrayLike, labels: ArrayLike) -> tuple[float, float, float]:
    tp, fp, _, fn = confusion(pred, labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_difference(
    a_successes: int, a_trials: int, b_successes: int, b_trials: int, z: float = 1.96
) -> tuple[float, float, float]:
    """Difference of two proportions ``a - b`` with Newcombe's hybrid score interval."""
    p_a = a_successes / a_trials if a_trials else 0.0
    p_b = b_successes / b_trials if b_trials else 0.0
    la, ua = wilson_interval(a_successes, a_trials, z)
    lb, ub = wilson_interval(b_successes, b_trials, z)
    diff = p_a - p_b
    low = diff - math.sqrt((p_a - la) ** 2 + (ub - p_b) ** 2)
    high = diff + math.sqrt((ua - p_a) ** 2 + (p_b - lb) ** 2)
    return diff, max(-1.0, low), min(1.0, high)
