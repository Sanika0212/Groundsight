"""Root-cause triage: why did a flagged claim go wrong, and where should the fix go?

The rules combine three signals per claim: which chunks support or contradict it, how
topically related the closest chunk is, and (when a reference answer is given) whether
the context contained the right answer at all. The cause is a *hypothesis*;
:mod:`groundsight.verify` confirms or rejects it by replaying the answer with the
matching fix applied.

==========================  ==============================================================
cause                       evidence pattern
==========================  ==============================================================
``retrieval_miss``          nothing in the context covers the claim (or the right answer)
``misleading_source``       a passage supports the claim, but the claim contradicts the
                            reference: the passage itself is wrong
``conflicting_sources``     one passage supports the claim and another contradicts it
``misread_context``         a passage contradicts the claim; the model got it wrong
``unsupported_addition``    the context covers the topic but never states the claim
``buried_evidence``         the needed passage sits in the middle of a long context
==========================  ==============================================================
"""

from __future__ import annotations

from groundsight.checkers import Checker
from groundsight.model import Audit, Cause, ClaimCheck, Diagnosis, Verdict
from groundsight.text import windows

FIXES = {
    Cause.RETRIEVAL_MISS: (
        "Check that the corpus contains this information, then the retriever: query "
        "rewriting, top-k, chunk size or the embedding model."
    ),
    Cause.MISLEADING_SOURCE: (
        "The answer faithfully repeats a passage that is itself wrong. Correct or remove "
        "the source document."
    ),
    Cause.CONFLICTING_SOURCES: (
        "Retrieved passages disagree. Deduplicate or version the corpus, or ask the model "
        "to surface conflicts instead of picking one."
    ),
    Cause.MISREAD_CONTEXT: (
        "The right information was in the context and the model contradicted it. Tighten "
        "the prompt, lower the temperature or use a stronger model."
    ),
    Cause.UNSUPPORTED_ADDITION: (
        "The passages cover this topic but don't state the claim. Instruct the model to "
        "answer only from the context, or require citations."
    ),
    Cause.BURIED_EVIDENCE: (
        "The needed passage sits in the middle of a long context. Rerank so the key "
        "passages come first, or pass fewer passages."
    ),
}


def is_middle(position: int, total: int) -> bool:
    """Middle of a long context: not among the first two or the last passage."""
    return total >= 5 and 2 <= position <= total - 2


def _label(audit: Audit, chunk_id: str) -> str:
    position = next(i for i, c in enumerate(audit.chunks) if c.id == chunk_id)
    return f"{chunk_id} (passage {position + 1} of {len(audit.chunks)})"


def _answer_chunks(audit: Audit, checker: Checker | None) -> list[str] | None:
    """Chunks that support the reference answer; ``None`` without a reference."""
    if not audit.reference or checker is None:
        return None
    pairs, owners = [], []
    for chunk in audit.chunks:
        for win in windows(chunk.text):
            pairs.append((win.text, audit.reference))
            owners.append(chunk.id)
    scores = checker.score(pairs) if pairs else []
    supported = {
        cid for cid, (s, _) in zip(owners, scores, strict=True) if s >= audit.thresholds.support
    }
    return [c.id for c in audit.chunks if c.id in supported]


def _diagnose_claim(
    claim: ClaimCheck,
    audit: Audit,
    answer_chunks: list[str] | None,
    relevance_threshold: float,
) -> Diagnosis:
    limits = audit.thresholds
    order = {c.id: i for i, c in enumerate(audit.chunks)}
    total = len(audit.chunks)
    supporters = [cid for cid, (s, _) in claim.per_chunk.items() if s >= limits.support]
    contradictors = [
        cid for cid, (s, c) in claim.per_chunk.items() if c >= limits.contradiction and c > s
    ]
    buried = [cid for cid in (answer_chunks or []) if is_middle(order[cid], total)]

    def make(cause: Cause, reason: str, chunk_ids: list[str]) -> Diagnosis:
        return Diagnosis(cause=cause, reason=reason, fix=FIXES[cause], chunk_ids=chunk_ids)

    if claim.verdict is Verdict.SUPPORTED and claim.reference_conflict:
        named = ", ".join(_label(audit, c) for c in supporters)
        if answer_chunks:
            return make(
                Cause.CONFLICTING_SOURCES,
                f"{named} states the claim, but {_label(audit, answer_chunks[0])} supports "
                "the reference answer instead.",
                supporters + answer_chunks,
            )
        return make(
            Cause.MISLEADING_SOURCE,
            f"The claim repeats {named}, but it contradicts the reference answer.",
            supporters,
        )
    if claim.verdict is Verdict.CONTRADICTED:
        if supporters:
            return make(
                Cause.CONFLICTING_SOURCES,
                f"{_label(audit, supporters[0])} supports the claim while "
                f"{_label(audit, contradictors[0])} contradicts it.",
                supporters + contradictors,
            )
        culprit = (
            contradictors[0]
            if contradictors
            else max(claim.per_chunk, key=lambda cid: claim.per_chunk[cid][1])
        )
        if is_middle(order[culprit], total):
            return make(
                Cause.BURIED_EVIDENCE,
                f"{_label(audit, culprit)} contradicts the claim, and it sits in the middle "
                "of a long context where models attend least.",
                [culprit],
            )
        return make(
            Cause.MISREAD_CONTEXT,
            f"{_label(audit, culprit)} contradicts the claim.",
            [culprit],
        )
    # Unsupported (or reference conflict without any supporting passage).
    if answer_chunks is not None:
        if not answer_chunks:
            return make(
                Cause.RETRIEVAL_MISS,
                "No passage supports the claim, and none contains the reference answer.",
                [],
            )
        if buried:
            return make(
                Cause.BURIED_EVIDENCE,
                f"The answer is in {_label(audit, buried[0])}, in the middle of a long "
                "context, but the claim isn't supported by any passage.",
                buried,
            )
        return make(
            Cause.UNSUPPORTED_ADDITION,
            f"The context contains the answer ({_label(audit, answer_chunks[0])}), but the "
            "claim goes beyond it.",
            answer_chunks,
        )
    if claim.relevance >= relevance_threshold:
        closest = max(claim.per_chunk, key=lambda cid: claim.per_chunk[cid][0])
        return make(
            Cause.UNSUPPORTED_ADDITION,
            f"{_label(audit, closest)} covers this topic but never states the claim.",
            [closest],
        )
    return make(
        Cause.RETRIEVAL_MISS,
        "No passage covers this claim; the context is about something else.",
        [],
    )


def diagnose(
    audit: Audit, *, checker: Checker | None = None, relevance_threshold: float = 0.5
) -> Audit:
    """Attach a :class:`Diagnosis` to every flagged claim of ``audit`` (in place)."""
    answer_chunks = _answer_chunks(audit, checker)
    for claim in audit.claims:
        claim.diagnosis = (
            _diagnose_claim(claim, audit, answer_chunks, relevance_threshold)
            if claim.flagged and audit.chunks
            else None
        )
    return audit
