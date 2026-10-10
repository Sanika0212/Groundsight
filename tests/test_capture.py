import asyncio
from pathlib import Path

import httpx
import pytest

import groundsight.capture as gs
from groundsight.model import Thresholds, Verdict

LIMITS = Thresholds(support=0.6, contradiction=0.5)
DOCS = ["Refunds are accepted within 30 days of delivery.", "Shipping is free over $50."]


class Document:
    """Minimal LangChain-style document."""

    def __init__(self, text: str, **metadata: object) -> None:
        self.page_content = text
        self.metadata = metadata


def test_decorators_record_and_audit(tmp_path: Path) -> None:
    @gs.retriever
    def search(query: str) -> list[Document]:
        return [Document(DOCS[0], source="policy.md"), Document(DOCS[1], id="ship-1")]

    @gs.generator
    def respond(query: str, docs: list[Document]) -> str:
        return "Refunds are accepted within 30 days of delivery. Returns are free forever."

    store = gs.Store(tmp_path / "audits.jsonl")
    with gs.trace("Refund window?", store=store, checker="lexical", thresholds=LIMITS) as t:
        respond(t.question, search(t.question))  # type: ignore[arg-type]
    assert [c.id for c in t.chunks] == ["c0", "ship-1"]
    assert t.chunks[0].source == "policy.md"
    assert t.audit is not None
    assert [c.verdict for c in t.audit.claims] == [Verdict.SUPPORTED, Verdict.UNSUPPORTED]
    stored = store.all()
    assert len(stored) == 1
    assert store.get(t.audit.id[:6]).answer == t.answer
    # Outside a trace the decorators are transparent.
    assert search("x")[0].page_content == DOCS[0]
    assert gs.current() is None


def test_store_keeps_latest_record_and_reports_ambiguity(tmp_path: Path) -> None:
    store = gs.Store(tmp_path / "s.jsonl")
    assert store.all() == []
    with gs.trace("q", store=store, checker="lexical") as t:
        t.retrieved(DOCS)
        t.answered("Shipping is free over $50.")
    assert t.audit is not None
    t.audit.meta["note"] = "updated"
    store.save(t.audit)
    assert len(store.all()) == 1
    assert store.all()[0].meta["note"] == "updated"
    with pytest.raises(KeyError, match="matches 0 audits"):
        store.get("zzz")


def test_trace_without_answer_or_audit(tmp_path: Path) -> None:
    with gs.trace("q", store=tmp_path / "s.jsonl") as t:
        t.retrieved(["a passage"])
    assert t.audit is None
    with gs.trace("q", audit=False) as t2:
        t2.answered("Something.")
    assert t2.audit is None
    second = gs.Trace()
    second.retrieved(["one"])
    assert [c.id for c in second.retrieved(["two", object()])] == ["c1", "c2"]


def chat_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/messages"):
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": "Refunds: 30 days."}]}
        )
    return httpx.Response(
        200, json={"choices": [{"message": {"content": "Shipping is free over $50."}}]}
    )


def test_capture_http_records_chat_answers(tmp_path: Path) -> None:
    assert "httpx" in gs.capture_http()
    try:
        client = httpx.Client(transport=httpx.MockTransport(chat_handler))
        with gs.trace("Shipping?", store=None, checker="lexical", thresholds=LIMITS) as t:
            t.retrieved(DOCS)
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert t.answer == "Shipping is free over $50."
        assert t.model == "gpt-x"
        assert t.messages == [{"role": "user", "content": "hi"}]
        assert t.audit is not None
        assert t.audit.meta == {"model": "gpt-x"}

        async def call() -> gs.Trace:
            aclient = httpx.AsyncClient(transport=httpx.MockTransport(chat_handler))
            with gs.trace("Refunds?", audit=False) as tr:
                await aclient.post("https://api.anthropic.com/v1/messages", json={"model": "m"})
                await aclient.post("https://example.com/other", json={})
            return tr

        assert asyncio.run(call()).answer == "Refunds: 30 days."
        client.post("https://api.openai.com/v1/chat/completions", json={})  # no trace: ignored
    finally:
        gs.release_http()
    assert gs._patched == {}
