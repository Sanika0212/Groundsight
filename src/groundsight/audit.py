"""Audit an answer against its sources, claim by claim."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from groundsight.calibrate import bundled_profile
from groundsight.checkers import Checker, load_checker, overlap
from groundsight.model import Audit, Chunk, ClaimCheck, Evidence, Thresholds, Verdict
from groundsight.text import Segment, sentences, windows

DEFAULT_CHECKER = "minicheck+nli"
_checkers: dict[str, Checker] = {}

ChunkLike = Chunk | str | Mapping[str, Any]


def as_chunks(chunks: Sequence[ChunkLike]) -> list[Chunk]:
    """Accept chunks as :class:`Chunk` objects, plain strings or dicts with ``text``."""
    out = []
    for i, item in enumerate(chunks):
        if isinstance(item, Chunk):
            out.append(item)
        elif isinstance(item, str):
            out.append(Chunk(id=f"c{i}", text=item))
        else:
            data = dict(item)
            text = str(data.pop("text", data.pop("page_content", data.pop("content", ""))))
            chunk_id = str(data.pop("id", data.pop("chunk_id", f"c{i}")))
            known = {k: data.pop(k) for k in ("score", "source") if k in data}
            out.append(Chunk(id=chunk_id, text=text, metadata=data, **known))
    return out


def get_checker(checker: Checker | str) -> Checker:
    """Resolve a checker name, reusing loaded models across audits."""
    if not isinstance(checker, str):
        return checker
    if checker not in _checkers:
        _checkers[checker] = load_checker(checker)
    return _checkers[checker]


def resolve_thresholds(
    checker: Checker, thresholds: Thresholds | None, target_recall: float | None
) -> Thresholds:
    if thresholds is not None:
        return thresholds
    profile = bundled_profile(checker.name)
    if profile is None:
        return Thresholds(calibration="default 0.5 (no calibration profile for this checker)")
    if target_recall is not None:
        return profile.conformal(target_recall)
    return profile.balanced()


def _audit_id(answer: str, chunks: Sequence[Chunk], question: str | None) -> str:
    digest = hashlib.sha256()
    for part in (question or "", answer, *(c.id + "\x00" + c.text for c in chunks)):
        digest.update(part.encode())
        digest.update(b"\x01")
    return digest.hexdigest()[:16]


def audit(
    answer: str,
    chunks: Sequence[ChunkLike],
    *,
    question: str | None = None,
    reference: str | None = None,
    checker: Checker | str = DEFAULT_CHECKER,
    thresholds: Thresholds | None = None,
    target_recall: float | None = None,
    max_words: int = 250,
    min_claim_words: int = 3,
    joint: bool = True,
    diagnose: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> Audit:
    """Check every claim in ``answer`` against ``chunks``.

    Args:
        answer: the generated answer.
        chunks: the passages the model was given, in prompt order.
        question: the user's question (used in reports and triage).
        reference: a known-good answer, if you have one. Claims that contradict it are
            flagged even when a source supports them (the source itself may be wrong).
        checker: a checker or a name for :func:`groundsight.checkers.load_checker`.
        thresholds: explicit thresholds; by default they come from the checker's
            bundled calibration profile.
        target_recall: with a calibration profile, choose the threshold that flags at
            least this share of unsupported claims (split conformal guarantee).
        max_words: long chunks are checked in windows of at most this many words.
        min_claim_words: sentences shorter than this are not treated as claims.
        joint: also check each claim against all chunks together, so a claim that
            combines facts from two passages can be supported.
        diagnose: attach a probable cause to every flagged claim (see :mod:`triage`).
    """
    chunk_list = as_chunks(chunks)
    check = get_checker(checker)
    limits = resolve_thresholds(check, thresholds, target_recall)
    claims = sentences(answer, min_words=min_claim_words)

    chunk_windows = {c.id: windows(c.text, max_words=max_words) for c in chunk_list}
    joint_text = "\n\n".join(c.text for c in chunk_list)
    use_joint = joint and len(chunk_list) > 1 and len(joint_text.split()) <= int(max_words * 1.6)

    pairs: list[tuple[str, str]] = []
    index: list[tuple[int, str, int]] = []  # (claim, chunk id or "*", window)
    for ci, claim in enumerate(claims):
        for chunk in chunk_list:
            for wi, win in enumerate(chunk_windows[chunk.id]):
                pairs.append((win.text, claim.text))
                index.append((ci, chunk.id, wi))
        if use_joint:
            pairs.append((joint_text, claim.text))
            index.append((ci, "*", 0))
        if reference:
            pairs.append((reference, claim.text))
            index.append((ci, "reference", 0))
    scores = check.score(pairs) if pairs else []

    window_scores: dict[tuple[int, str, int], tuple[float, float]] = dict(
        zip(index, scores, strict=True)
    )
    results = []
    for ci, claim in enumerate(claims):
        per_chunk: dict[str, tuple[float, float]] = {}
        for chunk in chunk_list:
            wins = [window_scores[(ci, chunk.id, wi)] for wi in range(len(chunk_windows[chunk.id]))]
            if wins:
                per_chunk[chunk.id] = (max(s for s, _ in wins), max(c for _, c in wins))
        support = max((s for s, _ in per_chunk.values()), default=0.0)
        contradiction = max((c for _, c in per_chunk.values()), default=0.0)
        if use_joint:
            support = max(support, window_scores[(ci, "*", 0)][0])
        verdict = limits.verdict(support, contradiction)
        conflict = False
        if reference:
            ref_support, ref_contra = window_scores[(ci, "reference", 0)]
            conflict = limits.verdict(ref_support, ref_contra) is Verdict.CONTRADICTED
        results.append(
            ClaimCheck(
                index=ci,
                text=claim.text,
                start=claim.start,
                end=claim.end,
                support=support,
                contradiction=contradiction,
                verdict=verdict,
                per_chunk=per_chunk,
                relevance=max((overlap(c.text, claim.text) for c in chunk_list), default=0.0),
                reference_conflict=conflict,
            )
        )

    _attach_evidence(results, chunk_list, chunk_windows, window_scores, check)
    result = Audit(
        answer=answer,
        chunks=chunk_list,
        claims=results,
        checker=check.name,
        thresholds=limits,
        question=question,
        reference=reference,
        id=_audit_id(answer, chunk_list, question),
        meta=dict(meta or {}),
    )
    if diagnose:
        from groundsight.triage import diagnose as run_triage

        run_triage(result, checker=check)
    return result


def _attach_evidence(
    claims: list[ClaimCheck],
    chunks: list[Chunk],
    chunk_windows: Mapping[str, list[Segment]],
    window_scores: Mapping[tuple[int, str, int], tuple[float, float]],
    check: Checker,
) -> None:
    """Point each claim at the sentence that best supports or contradicts it.

    The best window is refined to its best single sentence when that sentence carries
    most of the window's score; otherwise (evidence spread over several sentences) the
    whole window is the evidence.
    """
    by_id = {c.id: c for c in chunks}
    picks: list[tuple[ClaimCheck, str, Segment, int]] = []  # claim, chunk, window, column
    for claim in claims:
        column = 1 if claim.verdict is Verdict.CONTRADICTED else 0
        best: tuple[float, str, int] | None = None
        for chunk_id, wins in chunk_windows.items():
            for wi in range(len(wins)):
                value = window_scores[(claim.index, chunk_id, wi)][column]
                if best is None or value > best[0]:
                    best = (value, chunk_id, wi)
        if best is not None:
            picks.append((claim, best[1], chunk_windows[best[1]][best[2]], column))

    pairs = []
    owners = []
    for claim, _chunk_id, win, _ in picks:
        for sent in sentences(win.text):
            pairs.append((sent.text, claim.text))
            owners.append((claim.index, sent))
    sentence_scores = check.score(pairs) if pairs else []

    for claim, chunk_id, win, column in picks:
        chunk = by_id[chunk_id]
        window_value = window_scores[(claim.index, chunk_id, chunk_windows[chunk_id].index(win))]
        start, end, score = win.start, win.end, window_value
        candidates = [
            (sent, sc)
            for (owner, sent), sc in zip(owners, sentence_scores, strict=True)
            if owner == claim.index
        ]
        if candidates:
            sent, sc = max(candidates, key=lambda item: item[1][column])
            if sc[column] >= 0.8 * window_value[column]:
                start, end, score = win.start + sent.start, win.start + sent.end, sc
        claim.evidence = Evidence(
            chunk_id=chunk_id,
            start=start,
            end=end,
            text=chunk.text[start:end],
            support=score[0],
            contradiction=score[1],
        )
