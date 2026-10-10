"""Export audits as OpenTelemetry traces (OTLP JSON), one evaluation event per claim.

Each audit becomes a ``groundsight.audit`` span. Every claim adds a
``gen_ai.evaluation.result`` span event carrying the GenAI semantic-convention
attributes ``gen_ai.evaluation.name``, ``.score.value``, ``.score.label`` and
``.explanation``. Those conventions are in Development status and now live in the
``open-telemetry/semantic-conventions-genai`` repository, so names may still change.

Pass ``trace_id``/``parent_span_id`` to attach the audit to the span of the model call
it evaluates. No OpenTelemetry SDK is needed: OTLP/HTTP accepts plain JSON.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx

from groundsight._version import __version__
from groundsight.model import Audit

EVALUATION_EVENT = "gen_ai.evaluation.result"
EVALUATION_NAME = "groundsight.claim_grounding"


def _value(v: Any) -> dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def _attrs(values: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _value(v)} for k, v in values.items() if v is not None]


def _hex(seed: str, n: int) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:n]


def to_otlp(
    audit: Audit,
    *,
    service: str = "groundsight",
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    response_id: str | None = None,
) -> dict[str, Any]:
    start = int(audit.created * 1e9)
    events = []
    for claim in audit.claims:
        explanation = claim.diagnosis.reason if claim.diagnosis else None
        events.append(
            {
                "timeUnixNano": str(start),
                "name": EVALUATION_EVENT,
                "attributes": _attrs(
                    {
                        "gen_ai.evaluation.name": EVALUATION_NAME,
                        "gen_ai.evaluation.score.value": round(claim.support, 6),
                        "gen_ai.evaluation.score.label": claim.verdict.value,
                        "gen_ai.evaluation.explanation": explanation,
                        "gen_ai.response.id": response_id,
                        "groundsight.claim.index": claim.index,
                        "groundsight.claim.text": claim.text,
                        "groundsight.claim.contradiction": round(claim.contradiction, 6),
                        "groundsight.claim.cause": (
                            claim.diagnosis.cause.value if claim.diagnosis else None
                        ),
                        "groundsight.claim.evidence_chunk": (
                            claim.evidence.chunk_id if claim.evidence else None
                        ),
                    }
                ),
            }
        )
    counts = audit.counts()
    span: dict[str, Any] = {
        "traceId": trace_id or _hex(audit.id + "trace", 32),
        "spanId": _hex(audit.id + "span", 16),
        "name": "groundsight.audit",
        "kind": 1,
        "startTimeUnixNano": str(start),
        "endTimeUnixNano": str(start),
        "attributes": _attrs(
            {
                "groundsight.audit.id": audit.id,
                "groundsight.checker": audit.checker,
                "groundsight.groundedness": round(audit.groundedness, 6),
                "groundsight.claims": len(audit.claims),
                "groundsight.claims.unsupported": counts["unsupported"],
                "groundsight.claims.contradicted": counts["contradicted"],
                "groundsight.threshold.support": audit.thresholds.support,
                "groundsight.threshold.target_recall": audit.thresholds.target_recall,
            }
        ),
        "events": events,
        "status": {"code": 1},
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs({"service.name": service})},
                "scopeSpans": [
                    {"scope": {"name": "groundsight", "version": __version__}, "spans": [span]}
                ],
            }
        ]
    }


def write_otlp(audit: Audit, path: str | Path, **kwargs: Any) -> Path:
    out = Path(path)
    out.write_text(json.dumps(to_otlp(audit, **kwargs), indent=1))
    return out


def send_otlp(
    audit: Audit,
    endpoint: str = "http://localhost:4318/v1/traces",
    *,
    client: httpx.Client | None = None,
    **kwargs: Any,
) -> int:
    """POST the audit to an OTLP/HTTP collector; returns the HTTP status code."""
    http = client or httpx.Client(timeout=10.0)
    return http.post(endpoint, json=to_otlp(audit, **kwargs)).status_code
