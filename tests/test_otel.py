import json
from pathlib import Path

import httpx

from groundsight.audit import audit
from groundsight.model import Thresholds
from groundsight.otel import EVALUATION_EVENT, send_otlp, to_otlp, write_otlp


def make():  # type: ignore[no-untyped-def]
    return audit(
        "The tower is 500 metres tall. The tower is in Paris.",
        ["The tower is 330 metres tall and stands in Paris."],
        checker="lexical",
        thresholds=Thresholds(support=0.6),
    )


def attrs(item: dict) -> dict:
    return {a["key"]: next(iter(a["value"].values())) for a in item["attributes"]}


def test_audit_exports_one_evaluation_event_per_claim(tmp_path: Path) -> None:
    result = make()
    data = to_otlp(result, trace_id="a" * 32, parent_span_id="b" * 16, response_id="resp-1")
    span = data["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["traceId"] == "a" * 32
    assert span["parentSpanId"] == "b" * 16
    assert attrs(span)["groundsight.claims.contradicted"] == "1"
    events = span["events"]
    assert [e["name"] for e in events] == [EVALUATION_EVENT] * 2
    first, second = (attrs(e) for e in events)
    assert first["gen_ai.evaluation.name"] == "groundsight.claim_grounding"
    assert first["gen_ai.evaluation.score.label"] == "contradicted"
    assert first["gen_ai.response.id"] == "resp-1"
    assert first["groundsight.claim.cause"] == "misread_context"
    assert "gen_ai.evaluation.explanation" not in second  # supported: nothing to explain
    path = write_otlp(result, tmp_path / "audit.otlp.json")
    assert json.loads(path.read_text())["resourceSpans"]
    default = to_otlp(result)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert len(default["traceId"]) == 32
    assert "parentSpanId" not in default


def test_send_otlp_posts_json() -> None:
    seen: list[httpx.Request] = []
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(200))
    )
    assert send_otlp(make(), "http://collector:4318/v1/traces", client=client) == 200
    assert "resourceSpans" in json.loads(seen[0].content)
