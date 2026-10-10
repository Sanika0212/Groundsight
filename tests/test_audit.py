import pytest

from groundsight.audit import as_chunks, audit, get_checker, resolve_thresholds
from groundsight.checkers import LexicalChecker
from groundsight.model import Audit, Cause, Chunk, Thresholds, Verdict
from groundsight.triage import diagnose, is_middle

TOWER = "The Eiffel Tower is 330 metres tall. It was completed in 1889 for the World's Fair."
LIMITS = Thresholds(support=0.6, contradiction=0.5)


def run(answer: str, chunks, **kwargs) -> Audit:  # type: ignore[no-untyped-def]
    return audit(answer, chunks, checker="lexical", thresholds=LIMITS, **kwargs)


def test_claims_get_verdicts_evidence_and_offsets() -> None:
    answer = (
        "The Eiffel Tower was completed in 1889. The Eiffel Tower is 500 metres tall. "
        "Bananas grow in tropical climates."
    )
    result = run(answer, [TOWER, "Paris is the capital of France."], question="Tell me")
    verdicts = [c.verdict for c in result.claims]
    assert verdicts == [Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.UNSUPPORTED]
    assert all(answer[c.start : c.end] == c.text for c in result.claims)
    first = result.claims[0]
    assert first.evidence is not None
    assert first.evidence.chunk_id == "c0"
    # "It was completed..." alone doesn't name the tower, so the whole passage is evidence.
    assert first.evidence.text == TOWER
    assert TOWER[first.evidence.start : first.evidence.end] == first.evidence.text
    assert result.claims[1].evidence.text == "The Eiffel Tower is 330 metres tall."  # type: ignore[union-attr]
    assert result.groundedness == pytest.approx(1 / 3)
    assert result.counts() == {"supported": 1, "contradicted": 1, "unsupported": 1}
    assert [c.index for c in result.flagged] == [1, 2]
    assert len(result.id) == 16
    assert result.chunk("c1").text.startswith("Paris")


def test_joint_check_supports_claims_spanning_two_passages() -> None:
    chunks = ["Marie Curie was born in Warsaw.", "She won two Nobel prizes."]
    answer = "Marie Curie, born in Warsaw, won two Nobel prizes."
    assert run(answer, chunks).claims[0].verdict is Verdict.SUPPORTED
    assert run(answer, chunks, joint=False).claims[0].verdict is not Verdict.SUPPORTED


def test_round_trip_and_chunk_inputs() -> None:
    chunks = as_chunks(
        [
            Chunk("a", TOWER),
            {"id": "b", "text": "Paris is in France.", "score": 0.4, "source": "wiki", "x": 1},
            {"page_content": "LangChain style document."},
        ]
    )
    assert [c.id for c in chunks] == ["a", "b", "c2"]
    assert chunks[1].score == 0.4
    assert chunks[1].metadata == {"x": 1}
    result = run("The Eiffel Tower is 500 metres tall. Paris is in France.", chunks)
    again = Audit.from_dict(result.to_dict())
    assert again.to_dict() == result.to_dict()
    assert again.claims[0].diagnosis is not None
    assert again.claims[0].diagnosis.cause is Cause.MISREAD_CONTEXT


def test_empty_answer_and_no_chunks() -> None:
    assert run("", [TOWER]).claims == []
    assert run("", [TOWER]).groundedness == 1.0
    nothing = run("The Eiffel Tower is tall indeed.", [])
    assert nothing.claims[0].verdict is Verdict.UNSUPPORTED
    assert nothing.claims[0].diagnosis is None


def test_threshold_resolution() -> None:
    lexical = get_checker("lexical")
    assert get_checker("lexical") is lexical
    assert isinstance(get_checker(LexicalChecker()), LexicalChecker)
    assert resolve_thresholds(lexical, LIMITS, None) is LIMITS
    fallback = resolve_thresholds(lexical, None, 0.9)
    assert fallback.support == 0.5
    assert "no calibration profile" in (fallback.calibration or "")


def causes(result: Audit) -> list[Cause | None]:
    return [c.diagnosis.cause if c.diagnosis else None for c in result.claims]


def test_triage_without_reference() -> None:
    result = run(
        "The Eiffel Tower is 500 metres tall. "
        "The Eiffel Tower attracts tourists with nightly laser shows. "
        "Quantum computers use qubits for computation.",
        [TOWER, "The Eiffel Tower is in Paris and attracts tourists."],
    )
    assert causes(result) == [
        Cause.MISREAD_CONTEXT,
        Cause.UNSUPPORTED_ADDITION,
        Cause.RETRIEVAL_MISS,
    ]
    assert result.claims[0].diagnosis.chunk_ids == ["c0"]  # type: ignore[union-attr]
    assert "passage 1 of 2" in result.claims[0].diagnosis.reason  # type: ignore[union-attr]


def test_triage_with_reference_answer() -> None:
    wrong = "The Eiffel Tower is 300 metres tall."
    misleading = run(
        "The Eiffel Tower is 300 metres tall.",
        [wrong],
        reference="The Eiffel Tower is 330 metres tall.",
    )
    assert misleading.claims[0].verdict is Verdict.SUPPORTED
    assert misleading.claims[0].reference_conflict
    assert causes(misleading) == [Cause.MISLEADING_SOURCE]

    both = run(
        "The Eiffel Tower is 300 metres tall.",
        [wrong, "Official figures: the Eiffel Tower is 330 metres tall."],
        reference="The Eiffel Tower is 330 metres tall.",
    )
    assert causes(both) == [Cause.CONFLICTING_SOURCES]

    filler = [f"Paris has many museums, number {i}." for i in range(6)]
    context = [*filler[:3], "The tower opened to visitors in 1889.", *filler[3:]]
    buried = run("The tower opened to visitors in 1925.", context, reference="It opened in 1889.")
    assert causes(buried) == [Cause.BURIED_EVIDENCE]

    missing = run("The tower opened in 1925.", filler, reference="It opened in 1889.")
    assert causes(missing) == [Cause.RETRIEVAL_MISS]

    padded = run(
        "The tower opened to visitors in 1889 with fireworks and a grand parade.",
        ["The tower opened to visitors in 1889."],
        reference="It opened to visitors in 1889.",
    )
    assert causes(padded) == [Cause.UNSUPPORTED_ADDITION]


def test_conflicting_passages_without_reference() -> None:
    result = run(
        "The Eiffel Tower is 300 metres tall.",
        ["Some say the Eiffel Tower is 300 metres tall.", "The Eiffel Tower is 330 metres tall."],
    )
    claim = result.claims[0]
    assert claim.verdict is Verdict.CONTRADICTED or claim.diagnosis is None
    if claim.diagnosis is not None:
        assert claim.diagnosis.cause is Cause.CONFLICTING_SOURCES


def test_is_middle_and_rediagnosis() -> None:
    assert not is_middle(2, 4)
    assert [is_middle(i, 6) for i in range(6)] == [False, False, True, True, True, False]
    result = run("The Eiffel Tower is 500 metres tall.", [TOWER], diagnose=False)
    assert result.claims[0].diagnosis is None
    diagnose(result)
    assert causes(result) == [Cause.MISREAD_CONTEXT]
