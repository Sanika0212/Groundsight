"""Choosing the support threshold: best balanced accuracy, or a recall guarantee.

A *profile* stores, for one checker, the support scores of labelled claims from a
calibration set: claims annotators marked as hallucinated (positives) and claims they
accepted (negatives). From it:

* :meth:`Profile.balanced` picks the threshold with the best balanced accuracy.
* :meth:`Profile.conformal` picks the threshold that flags at least ``target_recall`` of
  hallucinated claims. By split conformal prediction, if new hallucinated claims are
  exchangeable with the calibration ones, a new hallucinated claim is flagged with
  probability at least ``target_recall`` (over the draw of the calibration set). The
  price of a higher target is more false alarms, which the profile also reports.

Groundsight ships profiles fitted on the RAGTruth training split; fit your own on
labelled examples from your domain with :func:`fit_profile` or ``groundsight
calibrate``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from groundsight.metrics import auroc, balanced_accuracy
from groundsight.model import Thresholds


def conformal_threshold(positive_scores: ArrayLike, target_recall: float) -> float:
    """Smallest threshold ``t`` such that flagging ``score < t`` has recall >= target.

    Uses the ``ceil((n + 1) * target)``-th smallest calibration score, the finite-sample
    correction of split conformal prediction. Returns ``inf`` (flag everything) when the
    calibration set is too small for the requested target.
    """
    if not 0 < target_recall < 1:
        raise ValueError("target_recall must be strictly between 0 and 1")
    scores = np.sort(np.asarray(positive_scores, dtype=float))
    n = len(scores)
    k = math.ceil((n + 1) * target_recall)
    if k > n:
        return math.inf
    return float(np.nextafter(scores[k - 1], math.inf))


def best_balanced_threshold(scores: ArrayLike, labels: ArrayLike) -> float:
    """Threshold maximising balanced accuracy of ``score < t`` as a detector of ``labels``."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    candidates = np.unique(np.concatenate([s, [s.max() + 1e-9 if len(s) else 1.0]]))
    best_t, best_ba = 0.5, -1.0
    for t in candidates:
        ba = balanced_accuracy(s < t, y)
        if ba > best_ba:
            best_t, best_ba = float(t), ba
    return best_t


@dataclass
class Profile:
    """Calibration scores for one checker on one labelled dataset."""

    checker: str
    source: str
    positives: list[float]  # support scores of claims labelled hallucinated
    negatives: list[float]  # support scores of claims labelled supported
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.positives) + len(self.negatives)

    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(self.positives + self.negatives, dtype=float)
        labels = np.asarray([True] * len(self.positives) + [False] * len(self.negatives))
        return scores, labels

    def auroc(self) -> float:
        scores, labels = self._arrays()
        return auroc(-scores, labels)

    def balanced(self) -> Thresholds:
        scores, labels = self._arrays()
        return Thresholds(
            support=best_balanced_threshold(scores, labels),
            calibration=f"{self.checker} on {self.source} (best balanced accuracy)",
        )

    def conformal(self, target_recall: float) -> Thresholds:
        return Thresholds(
            support=conformal_threshold(self.positives, target_recall),
            target_recall=target_recall,
            calibration=f"{self.checker} on {self.source} (conformal, recall >= {target_recall})",
        )

    def false_alarm_rate(self, thresholds: Thresholds) -> float:
        """Share of calibration claims labelled supported that the threshold would flag."""
        if not self.negatives:
            return 0.0
        return float(np.mean(np.asarray(self.negatives) < thresholds.support))

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        data = asdict(self)
        data["positives"] = [round(x, 5) for x in self.positives]
        data["negatives"] = [round(x, 5) for x in self.negatives]
        out.write_text(json.dumps(data) + "\n")
        return out

    @classmethod
    def load(cls, path: str | Path) -> Profile:
        return cls(**json.loads(Path(path).read_text()))


def fit_profile(
    checker: str, source: str, scores: ArrayLike, labels: ArrayLike, **meta: Any
) -> Profile:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    return Profile(checker, source, s[y].tolist(), s[~y].tolist(), dict(meta))


def bundled_profile(checker: str) -> Profile | None:
    """The calibration profile shipped with Groundsight for ``checker``, if any."""
    name = checker.replace("+", "_").replace(":", "_").replace("/", "_") + ".json"
    resource = resources.files("groundsight.profiles").joinpath(name)
    if not resource.is_file():
        return None
    return Profile(**json.loads(resource.read_text()))
