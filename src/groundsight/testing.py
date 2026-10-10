"""Assertions for RAG regression tests.

.. code-block:: python

    from groundsight.testing import assert_grounded

    def test_refund_answer():
        answer, docs = my_rag("What is the refund window?")
        assert_grounded(answer, docs, max_unsupported=0)

The checker comes from ``GROUNDSIGHT_CHECKER`` (default: the library default); set it
to ``lexical`` for fast, model-free checks in CI.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from groundsight.audit import DEFAULT_CHECKER, ChunkLike, audit
from groundsight.checkers import Checker
from groundsight.model import Audit


class GroundingError(AssertionError):
    def __init__(self, message: str, audit: Audit) -> None:
        super().__init__(message)
        self.audit = audit


def default_checker() -> str:
    return os.environ.get("GROUNDSIGHT_CHECKER", DEFAULT_CHECKER)


def explain(result: Audit) -> str:
    lines = [
        f"{result.groundedness:.0%} of {len(result.claims)} claims supported "
        f"(checker {result.checker}, support threshold {result.thresholds.support:.3f})"
    ]
    for claim in result.flagged:
        lines.append(f"  [{claim.verdict.value}] {claim.text}")
        if claim.diagnosis is not None:
            lines.append(f"      cause: {claim.diagnosis.cause.value}: {claim.diagnosis.reason}")
        if claim.evidence is not None and claim.verdict.value == "contradicted":
            lines.append(f"      evidence ({claim.evidence.chunk_id}): {claim.evidence.text}")
    return "\n".join(lines)


def assert_grounded(
    answer: str,
    chunks: Sequence[ChunkLike],
    *,
    question: str | None = None,
    reference: str | None = None,
    checker: Checker | str | None = None,
    max_unsupported: int = 0,
    max_contradicted: int = 0,
    min_groundedness: float | None = None,
    report: str | Path | None = None,
    **audit_options: Any,
) -> Audit:
    """Audit ``answer`` and fail with a readable explanation if it is not grounded enough.

    Returns the audit when the answer passes. With ``report``, an HTML report is written
    there when the assertion fails.
    """
    result = audit(
        answer,
        chunks,
        question=question,
        reference=reference,
        checker=checker or default_checker(),
        **audit_options,
    )
    counts = result.counts()
    problems = []
    if counts["contradicted"] > max_contradicted:
        problems.append(f"{counts['contradicted']} contradicted (allowed {max_contradicted})")
    if counts["unsupported"] > max_unsupported:
        problems.append(f"{counts['unsupported']} unsupported (allowed {max_unsupported})")
    if min_groundedness is not None and result.groundedness < min_groundedness:
        problems.append(f"groundedness {result.groundedness:.0%} < {min_groundedness:.0%}")
    if reference and any(c.reference_conflict for c in result.claims):
        problems.append("a claim contradicts the reference answer")
    if problems:
        message = "answer is not grounded: " + "; ".join(problems) + "\n" + explain(result)
        if report is not None:
            from groundsight.report import write_report

            message += f"\nreport: {write_report(result, report)}"
        raise GroundingError(message, result)
    return result
