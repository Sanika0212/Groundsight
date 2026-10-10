import os
from collections.abc import Sequence

import numpy as np
import pytest

from groundsight.checkers import (
    Combined,
    CosineChecker,
    HHEMChecker,
    LexicalChecker,
    MiniCheckChecker,
    NLIChecker,
    content_words,
    load_checker,
    numbers,
    overlap,
)
from groundsight.model import Thresholds, Verdict


def test_lexical_checker_spots_number_and_negation_conflicts() -> None:
    chunk = "The Eiffel Tower is 330 metres tall and was completed in 1889."
    check = LexicalChecker()
    (s_ok, c_ok), (s_num, c_num), (s_off, c_off) = check.score(
        [
            (chunk, "The Eiffel Tower was completed in 1889."),
            (chunk, "The Eiffel Tower is 500 metres tall."),
            (chunk, "Bananas are a good source of potassium."),
        ]
    )
    rules = Thresholds()
    assert rules.verdict(s_ok, c_ok) is Verdict.SUPPORTED
    assert rules.verdict(s_num, c_num) is Verdict.CONTRADICTED
    assert rules.verdict(s_off, c_off) is Verdict.UNSUPPORTED
    negated = check.score([("Aspirin is safe for adults.", "Aspirin is not safe for adults.")])
    assert rules.verdict(*negated[0]) is Verdict.CONTRADICTED


def test_word_helpers() -> None:
    assert content_words("The cat sat on the mat.") == {"cat", "sat", "mat"}
    assert numbers("It rose 2.5% to 1,200 units in 2024.") == {"2.5", "1200", "2024"}
    assert overlap("cats and dogs", "dogs") == 1.0
    assert overlap("anything", "the of and") == 0.0


class FakeNLI(NLIChecker):
    """NLI checker whose forward pass is scripted, so no weights are needed."""

    def __init__(self, logits: list[list[float]]) -> None:
        super().__init__("fake/nli-base", batch_size=2)
        self.calls = 0
        self._logits = logits
        self._labels = self.label_indices({0: "CONTRADICTION", 1: "entailment", 2: "neutral"})

    def _forward(self, batch: Sequence[tuple[str, str]]) -> np.ndarray:
        self.calls += 1
        return np.array([self._logits[int(c)] for _, c in batch], dtype=float)


def test_nli_mapping_softmax_cache_and_batches() -> None:
    nli = FakeNLI([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    pairs = [("p", "0"), ("p", "1"), ("p", "2"), ("p", "0")]
    scores = nli.score(pairs)
    assert scores[0][0] > 0.9
    assert scores[1][1] > 0.9
    assert max(scores[2]) < 0.1
    assert scores[0] == scores[3]
    assert nli.calls == 2  # three unique pairs in batches of two
    nli.score(pairs)
    assert nli.calls == 2  # cached
    assert nli.name == "nli"


def test_label_indices_from_model_config() -> None:
    nli = NLIChecker("cross-encoder/nli-deberta-v3-base")
    assert nli.label_indices({0: "contradiction", 1: "entailment", 2: "neutral"}) == (1, 0)
    assert nli.label_indices({"0": "ENTAILMENT", "1": "NEUTRAL", "2": "CONTRADICTION"}) == (0, 2)


def test_binary_checkers_convert_and_combine() -> None:
    mini = MiniCheckChecker()
    assert mini._convert(np.array([[0.0, 2.0]]))[0][0] == pytest.approx(0.8808, abs=1e-3)
    hhem = HHEMChecker()
    assert hhem._convert(np.array([[0.25]])) == [(0.25, 0.0)]
    cos = CosineChecker()
    assert cos._convert(np.array([[1.4], [-0.2]])) == [(1.0, 0.0), (0.0, 0.0)]

    class Scripted(LexicalChecker):
        def __init__(self, value: tuple[float, float], name: str) -> None:
            self.value, self.name = value, name

        def score(self, pairs):  # type: ignore[no-untyped-def]
            return [self.value for _ in pairs]

    both = Combined(Scripted((0.9, 0.0), "a"), Scripted((0.1, 0.7), "b"))
    assert both.name == "a+b"
    assert both.score([("p", "c")]) == [(0.9, 0.7)]


def test_load_checker() -> None:
    assert load_checker("lexical").name == "lexical"
    assert load_checker("minicheck+nli").name == "minicheck+nli"
    assert load_checker("nli-large").name.startswith("nli:")
    assert load_checker("cosine").name == "cosine"
    assert load_checker("hhem").name == "hhem"
    with pytest.raises(ValueError, match="unknown checker"):
        load_checker("magic")


@pytest.mark.models
@pytest.mark.skipif(os.environ.get("GROUNDSIGHT_TEST_MODELS") != "1", reason="downloads weights")
def test_real_nli_flags_a_contradiction() -> None:  # pragma: no cover - needs weights
    nli = load_checker("nli", device="cpu")
    chunk = "The Eiffel Tower is 330 metres tall."
    ((s, c),) = nli.score([(chunk, "The Eiffel Tower is 500 metres tall.")])
    assert Thresholds().verdict(s, c) is Verdict.CONTRADICTED
