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


def test_reference_answer_outranks_noisy_contradictions() -> None:
    """With a reference, where the answer is decides before passage contradictions do."""
    filler = [f"Paris has many museums, number {i}." for i in range(6)]
    # gold removed: a passage on a related topic contradicts the claim, but the context
    # never held the right answer, so the cause is the retrieval
    related = ["The Eiffel Tower is 330 metres tall.", *filler]
    gone = run("The tower opened in 1925.", related, reference="It opened in 1889.")
    assert causes(gone) == [Cause.RETRIEVAL_MISS]
    # a passage states the wrong claim and none has the reference: the source misled
    edited = run(
        "The tower opened to visitors in 1925.",
        ["The tower opened to visitors in 1925.", *filler],
        reference="The tower opened to visitors in 1889.",
    )
    assert causes(edited) == [Cause.MISLEADING_SOURCE]
    # the answer is only in the middle of a long context, even though the claim is
    # contradicted there: buried, not misread
    context = [*filler[:3], "The tower opened to visitors in 1889.", *filler[3:]]
    buried = run("The tower opened to visitors in 1925.", context, reference="It opened in 1889.")
    assert causes(buried) == [Cause.BURIED_EVIDENCE]
    # the answer is at the front and the claim contradicts it: misread
    front = run(
        "The tower opened to visitors in 1925.", context[3:], reference="It opened in 1889."
    )
    assert causes(front) in ([Cause.MISREAD_CONTEXT], [Cause.UNSUPPORTED_ADDITION])


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
    assert not is_middle(2, 5)
    assert [is_middle(i, 7) for i in range(7)] == [False, False, True, True, True, False, False]
    result = run("The Eiffel Tower is 500 metres tall.", [TOWER], diagnose=False)
    assert result.claims[0].diagnosis is None
    diagnose(result)
    assert causes(result) == [Cause.MISREAD_CONTEXT]


def test_contradiction_needs_a_passage_about_the_claim() -> None:
    class Contrary(LexicalChecker):
        """Says every pair contradicts, like an over-eager NLI model."""

        def score(self, pairs):  # type: ignore[no-untyped-def]
            return [(0.0, 0.99) for _ in pairs]

    related = audit(
        "The tower is 500 metres tall.",
        ["The tower is 330 metres tall."],
        checker=Contrary(),
        thresholds=LIMITS,
    )
    unrelated = audit(
        "Snapshots are stored in three data centres.",
        ["Snapshots are encrypted before they leave your device."],
        checker=Contrary(),
        thresholds=LIMITS,
    )
    assert related.claims[0].verdict is Verdict.CONTRADICTED
    assert unrelated.claims[0].verdict is Verdict.UNSUPPORTED


def test_evidence_can_be_skipped_for_speed() -> None:
    fast = run("The Eiffel Tower is 500 metres tall.", [TOWER], evidence=False)
    assert fast.claims[0].evidence is None
    assert fast.claims[0].verdict is Verdict.CONTRADICTED


def test_support_alone_decides_flags_unless_overrides_are_enabled() -> None:
    strict = Thresholds(support=0.6, contradiction=0.5)
    assert strict.verdict(0.9, 0.99) is Verdict.SUPPORTED
    assert strict.verdict(0.2, 0.7) is Verdict.CONTRADICTED
    assert strict.verdict(0.2, 0.1) is Verdict.UNSUPPORTED
    disputes = Thresholds(support=0.6, contradiction=0.5, contradiction_overrides_support=True)
    assert disputes.verdict(0.9, 0.99) is Verdict.CONTRADICTED
    assert disputes.verdict(0.9, 0.7) is Verdict.SUPPORTED  # contradiction must beat support


def test_restated_reference_means_a_misleading_source() -> None:
    filler = [f"Paris has many museums, number {i}." for i in range(5)]
    edited = "A 2002 comedy horror by Amberley Village."
    result = run(
        "Makdee was made by Amberley Village, a Mumbai director known for horror comedies.",
        [*filler, edited],
        question="Who directed Makdee?",
        reference="The 2002 comedy horror film Makdee was directed by Vishal Bhardwaj.",
    )
    claim = result.flagged[0]
    assert claim.diagnosis is not None
    assert claim.diagnosis.cause is Cause.MISLEADING_SOURCE
    assert claim.diagnosis.chunk_ids == ["c5"]
    # without a passage resembling the reference, it is a retrieval miss
    missing = run(
        "Makdee was made by Amberley Village, a Mumbai director known for horror comedies.",
        filler,
        question="Who directed Makdee?",
        reference="The 2002 comedy horror film Makdee was directed by Vishal Bhardwaj.",
    )
    assert causes(missing) == [Cause.RETRIEVAL_MISS]


def test_two_hop_reference_finds_conflicting_copies_and_buried_answers() -> None:
    filler = [f"Paris has many museums, number {i}." for i in range(6)]
    question = "Who directed the 2002 film produced by the studio in Mumbai?"
    reference = "Makdee, the 2002 comedy horror film, was directed by Vishal Bhardwaj."
    original = "Makdee is a comedy horror film from 2002 directed by Vishal Bhardwaj."
    copy = "Makdee is a comedy horror film from 2002 directed by Amberley Village."
    claim = (
        "Makdee, the 2002 comedy horror film, was directed by Amberley Village, a young "
        "newcomer with a small budget and unknown actors."
    )
    from groundsight.checkers import content_words

    class Strict(LexicalChecker):
        """Supports only when every content word of the claim is in the passage."""

        def score(self, pairs):  # type: ignore[no-untyped-def]
            return [(float(content_words(c) <= content_words(p)), 0.0) for p, c in pairs]

    def run(answer, chunks, **kwargs):  # type: ignore[no-untyped-def]
        return audit(answer, chunks, checker=Strict(), thresholds=LIMITS, **kwargs)

    both = run(claim, [original, *filler, copy], question=question, reference=reference)
    assert causes(both) == [Cause.CONFLICTING_SOURCES]
    assert both.flagged[0].diagnosis.chunk_ids == ["c7", "c0"]  # type: ignore[union-attr]
    middle = [*filler[:3], original, *filler[3:]]
    buried = run(claim, middle, question=question, reference=reference)
    assert causes(buried) == [Cause.BURIED_EVIDENCE]


def test_short_answers_are_checked_with_their_question() -> None:
    chunks = ["The founder was born in Selma, Alabama.", "Paris has many museums."]
    bare = run("Birmingham, Alabama", chunks)
    assert bare.claims == []  # two words: not a claim on its own
    asked = run("Birmingham, Alabama", chunks, question="Where was the founder born?")
    assert len(asked.claims) == 1
    claim = asked.claims[0]
    assert claim.text == "Birmingham, Alabama"
    assert (claim.start, claim.end) == (0, len("Birmingham, Alabama"))
    supported = run(
        "Birmingham, Alabama",
        ["The founder was born in Birmingham, Alabama."],
        question="Where was the founder born?",
    )
    assert supported.claims[0].verdict is Verdict.SUPPORTED
    two = run("Tulips bloom. Roses need pruning.", chunks, question="Flowers?")
    assert [c.text for c in two.claims] == ["Roses need pruning."]  # sentences stay sentences
