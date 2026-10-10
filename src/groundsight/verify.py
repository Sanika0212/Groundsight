"""Confirm a diagnosis by replaying the answer with the fix it implies.

For each flagged claim, the triage cause suggests an intervention on the model's input:
drop the misleading passage, move the evidence to the top, add the missing document,
or tighten the instruction. The answer is regenerated several times *with* the
intervention and several times *without* it (plain resampling, the noise floor). The
claim counts as fixed in a sample when the new answer no longer asserts it, or, if a
reference answer is known, when the new answer states the reference.

A cause is **confirmed** when the intervention's fix rate beats the resampling rate,
with a 95% Newcombe interval on the difference that excludes zero.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from groundsight.audit import get_checker
from groundsight.checkers import Checker
from groundsight.generate import STRICT_INSTRUCTION, Responder, build_messages
from groundsight.metrics import newcombe_difference, wilson_interval
from groundsight.model import Audit, Cause, Chunk, ClaimCheck

Kind = Literal["resample", "drop", "promote", "add", "instruct"]


@dataclass(frozen=True)
class Intervention:
    kind: Kind
    chunk_ids: tuple[str, ...] = ()
    text: str = ""

    @property
    def label(self) -> str:
        ids = ", ".join(self.chunk_ids)
        return {
            "resample": "resample (no change)",
            "drop": f"drop {ids}",
            "promote": f"move {ids} to the top",
            "add": "add the missing document",
            "instruct": "answer only from the context",
        }[self.kind]

    def apply(self, chunks: Sequence[Chunk]) -> tuple[list[Chunk], str | None]:
        """The chunks and instruction to answer with under this intervention."""
        items = list(chunks)
        if self.kind == "drop":
            return [c for c in items if c.id not in self.chunk_ids], None
        if self.kind == "promote":
            moved = [c for c in items if c.id in self.chunk_ids]
            return moved + [c for c in items if c.id not in self.chunk_ids], None
        if self.kind == "add":
            return [Chunk(id="added", text=self.text, source="intervention"), *items], None
        if self.kind == "instruct":
            return items, self.text or STRICT_INSTRUCTION
        return items, None


def interventions_for(claim: ClaimCheck, gold: str | None = None) -> list[Intervention]:
    """Interventions that would fix ``claim`` if its diagnosis is right."""
    if claim.diagnosis is None:
        return []
    cause, ids = claim.diagnosis.cause, tuple(claim.diagnosis.chunk_ids)
    supporters = tuple(cid for cid in ids if claim.per_chunk.get(cid, (0.0, 0.0))[0] >= 0.5)
    if cause is Cause.RETRIEVAL_MISS:
        if gold:
            return [Intervention("add", text=gold)]
        return [Intervention("instruct")]
    if cause is Cause.MISLEADING_SOURCE:
        return [Intervention("drop", ids)]
    if cause is Cause.CONFLICTING_SOURCES:
        return [Intervention("drop", supporters or ids[:1])]
    if cause in (Cause.MISREAD_CONTEXT, Cause.BURIED_EVIDENCE):
        out = [Intervention("promote", ids)] if ids else []
        return [*out, Intervention("instruct")]
    return [Intervention("instruct")]  # unsupported addition


@dataclass
class Trial:
    intervention: Intervention
    answers: list[str] = field(default_factory=list)
    fixed: list[bool] = field(default_factory=list)

    @property
    def successes(self) -> int:
        return sum(self.fixed)

    @property
    def samples(self) -> int:
        return len(self.fixed)

    @property
    def rate(self) -> float:
        return self.successes / self.samples if self.samples else 0.0

    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.samples)


@dataclass
class Verification:
    claim_index: int
    cause: Cause
    baseline: Trial
    trials: list[Trial]

    def effect(self, trial: Trial) -> tuple[float, float, float]:
        """Fix rate of ``trial`` minus the resampling rate, with a 95% interval."""
        return newcombe_difference(
            trial.successes, trial.samples, self.baseline.successes, self.baseline.samples
        )

    @property
    def best(self) -> Trial | None:
        return max(self.trials, key=lambda t: self.effect(t)[0], default=None)

    @property
    def confirmed(self) -> bool:
        return any(self.effect(t)[1] > 0 for t in self.trials)


def is_fixed(
    new_answer: str,
    claim: ClaimCheck,
    audit: Audit,
    checker: Checker,
) -> bool:
    """Whether a regenerated answer no longer has the problem ``claim`` had."""
    threshold = audit.thresholds.support
    if audit.reference:
        ((support, _),) = checker.score([(new_answer, audit.reference)])
        return support >= threshold
    ((support, _),) = checker.score([(new_answer, claim.text)])
    return support < threshold


def estimate(audit: Audit, *, samples: int = 5, gold: str | None = None) -> dict[str, int]:
    """Calls and rough prompt tokens a :func:`verify` run would use (ask before spending)."""
    calls = 0
    for claim in audit.flagged:
        n = len(interventions_for(claim, gold))
        if n:
            calls += samples * (n + 1)
    prompt = sum(len(m["content"]) for m in build_messages(audit.question, audit.chunks))
    return {"calls": calls, "prompt_tokens": calls * (prompt // 4 + 50)}


def verify(
    audit: Audit,
    responder: Responder,
    *,
    samples: int = 5,
    checker: Checker | str | None = None,
    gold: str | None = None,
    claims: Sequence[int] | None = None,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> list[Verification]:
    """Replay ``audit``'s answer under each flagged claim's fix and measure the effect.

    Diagnoses are updated in place: ``confidence`` becomes ``"confirmed"`` or
    ``"not confirmed"``.
    """
    check = get_checker(checker or audit.checker)
    chosen = [c for c in audit.flagged if claims is None or c.index in claims]
    results = []
    baseline_cache: dict[tuple[Kind, tuple[str, ...], str], list[str]] = {}

    def run(intervention: Intervention, claim: ClaimCheck) -> Trial:
        key = (intervention.kind, intervention.chunk_ids, intervention.text)
        if key not in baseline_cache:
            chunks, instruction = intervention.apply(audit.chunks)
            baseline_cache[key] = [
                responder(audit.question, chunks, instruction, seed + i) for i in range(samples)
            ]
        answers = baseline_cache[key]
        return Trial(intervention, answers, [is_fixed(a, claim, audit, check) for a in answers])

    for claim in chosen:
        options = interventions_for(claim, gold)
        if not options or claim.diagnosis is None:
            continue
        baseline = run(Intervention("resample"), claim)
        trials = [run(option, claim) for option in options]
        result = Verification(claim.index, claim.diagnosis.cause, baseline, trials)
        claim.diagnosis.confidence = "confirmed" if result.confirmed else "not confirmed"
        results.append(result)
        if progress is not None:
            best = result.best
            if best is not None:
                diff, low, high = result.effect(best)
                progress(
                    f"claim {claim.index}: {claim.diagnosis.cause.value} "
                    f"{claim.diagnosis.confidence}; {best.intervention.label} fixes "
                    f"{best.successes}/{best.samples} vs {baseline.successes}/{baseline.samples} "
                    f"by resampling ({diff:+.0%}, 95% CI {low:+.0%} to {high:+.0%})"
                )
    return results
