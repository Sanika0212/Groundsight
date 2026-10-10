"""Data model: chunks, claim checks, diagnoses and audits.

Everything is a plain dataclass that round-trips through JSON, so audits can be
stored, diffed, rendered offline and exported.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class Cause(StrEnum):
    """Why a claim went wrong, phrased as the part of the pipeline to fix."""

    RETRIEVAL_MISS = "retrieval_miss"
    MISLEADING_SOURCE = "misleading_source"
    CONFLICTING_SOURCES = "conflicting_sources"
    MISREAD_CONTEXT = "misread_context"
    UNSUPPORTED_ADDITION = "unsupported_addition"
    BURIED_EVIDENCE = "buried_evidence"


@dataclass(frozen=True)
class Chunk:
    """One retrieved passage, in the order it was given to the model."""

    id: str
    text: str
    score: float | None = None  # the retriever's own score, if any
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    """The span of a chunk that best supports or contradicts a claim."""

    chunk_id: str
    start: int
    end: int
    text: str
    support: float
    contradiction: float


@dataclass
class Diagnosis:
    cause: Cause
    reason: str
    fix: str
    chunk_ids: list[str] = field(default_factory=list)  # chunks implicated in the cause
    confidence: str = "hypothesis"  # "hypothesis" until a counterfactual check confirms it


@dataclass
class ClaimCheck:
    index: int
    text: str
    start: int
    end: int
    support: float
    contradiction: float
    verdict: Verdict
    evidence: Evidence | None = None
    per_chunk: dict[str, tuple[float, float]] = field(default_factory=dict)
    relevance: float = 0.0  # topical overlap with the closest chunk (not a grounding signal)
    reference_conflict: bool = False  # contradicts a supplied reference answer
    diagnosis: Diagnosis | None = None

    @property
    def flagged(self) -> bool:
        return self.verdict is not Verdict.SUPPORTED or self.reference_conflict


@dataclass(frozen=True)
class Thresholds:
    """Support at or above ``support`` counts as supported; contradiction at or above
    ``contradiction`` (and above the support score) counts as contradicted."""

    support: float = 0.5
    contradiction: float = 0.5
    target_recall: float | None = None  # set when ``support`` comes from conformal calibration
    calibration: str | None = None  # where the thresholds came from

    def verdict(self, support: float, contradiction: float) -> Verdict:
        contradicted = contradiction >= self.contradiction and contradiction > support
        if support >= self.support and not contradicted:
            return Verdict.SUPPORTED
        if contradicted:
            return Verdict.CONTRADICTED
        return Verdict.UNSUPPORTED


@dataclass
class Audit:
    answer: str
    chunks: list[Chunk]
    claims: list[ClaimCheck]
    checker: str
    thresholds: Thresholds
    question: str | None = None
    reference: str | None = None
    id: str = ""
    created: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def flagged(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.flagged]

    @property
    def groundedness(self) -> float:
        """Share of claims supported by the sources (1.0 for an answer with no claims)."""
        if not self.claims:
            return 1.0
        return sum(c.verdict is Verdict.SUPPORTED for c in self.claims) / len(self.claims)

    def counts(self) -> dict[str, int]:
        out = {v.value: 0 for v in Verdict}
        for c in self.claims:
            out[c.verdict.value] += 1
        return out

    def chunk(self, chunk_id: str) -> Chunk:
        return next(c for c in self.chunks if c.id == chunk_id)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for claim in data["claims"]:
            claim["per_chunk"] = {k: list(v) for k, v in claim["per_chunk"].items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Audit:
        claims = []
        for raw in data["claims"]:
            evidence = Evidence(**raw["evidence"]) if raw.get("evidence") else None
            diagnosis = None
            if raw.get("diagnosis"):
                d = dict(raw["diagnosis"])
                diagnosis = Diagnosis(**{**d, "cause": Cause(d["cause"])})
            claims.append(
                ClaimCheck(
                    **{
                        **raw,
                        "verdict": Verdict(raw["verdict"]),
                        "evidence": evidence,
                        "diagnosis": diagnosis,
                        "per_chunk": {k: (v[0], v[1]) for k, v in raw["per_chunk"].items()},
                    }
                )
            )
        return cls(
            answer=data["answer"],
            chunks=[Chunk(**c) for c in data["chunks"]],
            claims=claims,
            checker=data["checker"],
            thresholds=Thresholds(**data["thresholds"]),
            question=data.get("question"),
            reference=data.get("reference"),
            id=data.get("id", ""),
            created=data.get("created", 0.0),
            meta=data.get("meta", {}),
        )
