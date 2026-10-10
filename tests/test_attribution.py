from collections.abc import Sequence

import numpy as np
import pytest

from groundsight.attribution import Attribution, _ridge, attribute
from groundsight.audit import audit
from groundsight.generate import Message
from groundsight.model import Thresholds


class KeywordScorer:
    """Pretend LM: a claim is likely when a passage containing its key word is present.

    Each claim's log-probability is -5, plus 4 for every present passage that mentions
    the claim's first word, plus 1 for a passage that mentions "gardening".
    """

    def __init__(self) -> None:
        self.calls = 0

    def span_logprobs(
        self, messages: list[Message], answer: str, spans: Sequence[tuple[int, int]]
    ) -> list[float]:
        self.calls += 1
        context = messages[1]["content"].lower()
        passages = context.split("\n\n")
        out = []
        for s, e in spans:
            key = answer[s:e].split()[0].lower()
            value = -5.0 + 4.0 * sum(key in p for p in passages)
            value += 1.0 * any("gardening" in p for p in passages)
            out.append(value)
        return out


CHUNKS = [
    "Tulips bloom in spring across the Netherlands.",
    "General context about gardening and flowers.",
    "Roses need regular pruning to flower well.",
]
ANSWER = "Tulips bloom in spring. Roses need pruning."


def make():  # type: ignore[no-untyped-def]
    return audit(ANSWER, CHUNKS, checker="lexical", thresholds=Thresholds(), question="Flowers?")


def test_leave_one_out_finds_each_claims_source() -> None:
    scorer = KeywordScorer()
    result = attribute(make(), scorer)
    assert result.method == "loo"
    assert result.passes == 4 == scorer.calls
    assert result.top(0) == "c0"
    assert result.top(1) == "c2"
    ranked = result.for_claim(0)
    assert ranked[0] == ("c0", pytest.approx(4.0))
    assert ranked[1] == ("c1", pytest.approx(1.0))
    np.testing.assert_allclose(result.base, [0.0, 0.0])
    data = result.to_dict()
    assert data["chunk_ids"] == ["c0", "c1", "c2"]
    assert data["scores"][1][2] == pytest.approx(4.0)


def test_surrogate_recovers_additive_effects() -> None:
    result = attribute(make(), KeywordScorer(), method="surrogate", ablations=24, seed=3)
    assert result.method == "surrogate"
    assert result.passes == 25
    np.testing.assert_allclose(result.scores[0], [4.0, 1.0, 0.0], atol=0.05)
    np.testing.assert_allclose(result.scores[1], [0.0, 1.0, 4.0], atol=0.05)


def test_claim_subset_and_ridge() -> None:
    result = attribute(make(), KeywordScorer(), claims=[1])
    assert result.claim_indices == [1]
    assert isinstance(result, Attribution)
    x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
    y = 2 * x[:, 0] - x[:, 1] + 3
    np.testing.assert_allclose(_ridge(x, y, 1e-9), [2, -1], atol=1e-6)
