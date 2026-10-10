import math

import numpy as np
import pytest

from groundsight.calibrate import (
    Profile,
    best_balanced_threshold,
    bundled_profile,
    conformal_threshold,
    fit_profile,
)
from groundsight.metrics import (
    auroc,
    balanced_accuracy,
    confusion,
    newcombe_difference,
    precision_recall_f1,
    wilson_interval,
)


def test_metrics() -> None:
    assert auroc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert math.isnan(auroc([0.1], [True]))
    assert confusion([True, True, False, False], [True, False, False, True]) == (1, 1, 1, 1)
    assert balanced_accuracy([True, False], [True, False]) == 1.0
    assert precision_recall_f1([True, True], [True, False]) == (0.5, 1.0, pytest.approx(2 / 3))
    assert precision_recall_f1([False], [False]) == (0.0, 0.0, 0.0)
    low, high = wilson_interval(0, 3)
    assert low == 0.0
    assert 0.5 < high < 0.6
    assert wilson_interval(0, 0) == (0.0, 1.0)
    diff, lo, hi = newcombe_difference(9, 10, 1, 10)
    assert diff == pytest.approx(0.8)
    assert 0.3 < lo < diff < hi <= 1.0


def test_conformal_threshold_order_statistic() -> None:
    scores = [i / 10 for i in range(10)]  # 0.0 .. 0.9
    t = conformal_threshold(scores, 0.8)  # k = ceil(11 * 0.8) = 9 -> 0.8
    assert 0.8 < t < 0.8001
    assert np.mean(np.asarray(scores) < t) >= 0.8
    assert conformal_threshold(scores, 0.95) == math.inf  # needs more than 10 points
    with pytest.raises(ValueError, match="between 0 and 1"):
        conformal_threshold(scores, 1.0)


@pytest.mark.parametrize("target", [0.8, 0.9])
def test_conformal_recall_holds_on_average_over_calibration_draws(target: float) -> None:
    # The guarantee is marginal: averaged over calibration sets, fresh recall >= target.
    rng = np.random.default_rng(7)
    fresh = rng.beta(2, 5, size=20_000)
    recalls = [
        np.mean(fresh < conformal_threshold(rng.beta(2, 5, size=100), target)) for _ in range(300)
    ]
    assert np.mean(recalls) >= target - 0.005


def test_profile_thresholds_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    pos = rng.beta(2, 5, 300)  # hallucinated claims score low
    neg = rng.beta(5, 2, 700)
    scores = np.concatenate([pos, neg])
    labels = [True] * 300 + [False] * 700
    prof = fit_profile("toy", "synthetic", scores, labels, split="train")
    assert prof.n == 1000
    assert 0.85 < prof.auroc() < 1.0
    best = prof.balanced()
    assert 0.3 < best.support < 0.7
    assert best_balanced_threshold(scores, labels) == best.support
    strict = prof.conformal(0.95)
    assert strict.support > best.support
    assert strict.target_recall == 0.95
    assert prof.false_alarm_rate(strict) > prof.false_alarm_rate(best)
    path = prof.save(tmp_path / "toy.json")
    again = Profile.load(path)
    assert again.checker == "toy"
    assert again.meta == {"split": "train"}
    assert len(again.positives) == 300
    assert Profile("x", "y", [], []).false_alarm_rate(best) == 0.0


def test_missing_bundled_profile() -> None:
    assert bundled_profile("no-such-checker") is None
