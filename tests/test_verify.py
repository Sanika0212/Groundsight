import json
from collections.abc import Sequence
from dataclasses import replace

import httpx
import pytest

from groundsight.audit import audit, get_checker
from groundsight.generate import (
    STRICT_INSTRUCTION,
    AnthropicChat,
    GenerationError,
    OpenAIChat,
    PipelineResponder,
    PromptResponder,
    build_messages,
)
from groundsight.model import Cause, Chunk, Thresholds
from groundsight.verify import Intervention, estimate, interventions_for, is_fixed, verify

LIMITS = Thresholds(support=0.6, contradiction=0.5)
FILLER = [f"Paris has many museums, number {i}." for i in range(6)]
TRUE = "The Eiffel Tower is 330 metres tall."


def chat_reply(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


def test_prompt_and_openai_chat() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=chat_reply("ok"))

    chat = OpenAIChat(
        api_key="sk-test", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    messages = build_messages("Q?", [Chunk("a", "alpha"), Chunk("b", "beta")])
    assert messages[1]["content"] == "Context:\n[1] alpha\n\n[2] beta\n\nQuestion: Q?"
    assert chat(messages, temperature=0.3, seed=4) == "ok"
    body = json.loads(seen[0].content)
    assert body["seed"] == 4
    assert body["temperature"] == 0.3
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    assert (chat.calls, chat.input_tokens, chat.output_tokens) == (1, 7, 3)
    assert chat.name == "openai:gpt-4.1-mini"


def test_anthropic_chat_moves_system_prompt() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 2}},
        )

    chat = AnthropicChat(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert chat(build_messages("Q", [Chunk("a", "x")], "Be brief.")) == "hi"
    assert seen[0]["system"] == "Be brief."
    assert [m["role"] for m in seen[0]["messages"]] == ["user"]


def test_http_errors_retry_or_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    queue = [
        httpx.Response(429, headers={"retry-after": "1"}, json={"error": {"code": "rate_limit"}}),
        httpx.Response(200, json=chat_reply("done")),
        httpx.Response(429, json={"error": {"code": "insufficient_quota", "message": "No credit"}}),
        httpx.Response(400, text="not json"),
    ]
    client = httpx.Client(transport=httpx.MockTransport(lambda r: queue.pop(0)))
    chat = OpenAIChat(api_key="sk-secret", client=client, sleep=waits.append)
    assert chat([]) == "done"
    assert waits == [1.0]
    with pytest.raises(GenerationError, match="insufficient_quota: No credit") as info:
        chat([])
    assert "sk-secret" not in str(info.value)
    with pytest.raises(GenerationError, match="400"):
        chat([])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="OPENAI_API_KEY"):
        OpenAIChat()
    with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY"):
        AnthropicChat()


def test_interventions_apply() -> None:
    chunks = [Chunk("a", "A"), Chunk("b", "B"), Chunk("c", "C")]
    assert [c.id for c in Intervention("drop", ("b",)).apply(chunks)[0]] == ["a", "c"]
    assert [c.id for c in Intervention("promote", ("c",)).apply(chunks)[0]] == ["c", "a", "b"]
    added, _ = Intervention("add", text="gold").apply(chunks)
    assert added[0].text == "gold"
    assert Intervention("instruct").apply(chunks)[1] == STRICT_INSTRUCTION
    assert Intervention("resample").apply(chunks) == (chunks, None)
    assert Intervention("promote", ("c",)).label == "move c to the top"


def tower_pipeline(question, chunks: Sequence[Chunk], instruction, seed) -> str:  # type: ignore[no-untyped-def]
    """Reads only the first two passages, unless told to use only the context."""
    window = chunks if instruction else chunks[:2]
    if any("330" in c.text for c in window):
        return TRUE
    return "The Eiffel Tower is 500 metres tall."


def test_verify_confirms_buried_evidence_by_promoting_it() -> None:
    context = [*FILLER[:3], TRUE, *FILLER[3:]]
    result = audit(
        "The Eiffel Tower is 500 metres tall.", context, checker="lexical", thresholds=LIMITS
    )
    claim = result.claims[0]
    assert claim.diagnosis.cause is Cause.BURIED_EVIDENCE  # type: ignore[union-attr]
    options = interventions_for(claim)
    assert [o.kind for o in options] == ["promote", "instruct"]
    assert estimate(result, samples=3)["calls"] == 9
    lines: list[str] = []
    (check,) = verify(result, PipelineResponder(tower_pipeline), samples=3, progress=lines.append)
    assert check.baseline.rate == 0.0
    assert check.trials[0].rate == 1.0
    assert check.confirmed
    assert check.best is not None
    diff, low, _ = check.effect(check.best)
    assert diff == 1.0
    assert low > 0
    assert claim.diagnosis.confidence == "confirmed"  # type: ignore[union-attr]
    assert "confirmed; move c3 to the top fixes 3/3 vs 0/3" in lines[0]


def test_verify_rejects_a_diagnosis_the_fix_does_not_change() -> None:
    result = audit(
        "The Eiffel Tower is 500 metres tall.", [TRUE], checker="lexical", thresholds=LIMITS
    )
    stubborn = PipelineResponder(lambda q, c, i, s: "The Eiffel Tower is 500 metres tall.")
    (check,) = verify(result, stubborn, samples=2)
    assert not check.confirmed
    assert result.claims[0].diagnosis.confidence == "not confirmed"  # type: ignore[union-attr]


def test_fixes_with_reference_gold_and_prompt_responder() -> None:
    result = audit(
        "The tower opened in 1925.",
        FILLER,
        reference="It opened in 1889.",
        checker="lexical",
        thresholds=LIMITS,
    )
    claim = result.claims[0]
    assert claim.diagnosis.cause is Cause.RETRIEVAL_MISS  # type: ignore[union-attr]
    assert interventions_for(claim, gold="The tower opened in 1889.")[0].kind == "add"
    assert interventions_for(claim)[0].kind == "instruct"

    class Echo:
        name = "echo"

        def __call__(self, messages, *, temperature=0.0, seed=None):  # type: ignore[no-untyped-def]
            return "It opened in 1889." if "1889" in messages[1]["content"] else "Unknown."

    responder = PromptResponder(Echo(), temperature=0.5)
    (check,) = verify(result, responder, samples=2, gold="The tower opened in 1889.")
    assert check.confirmed
    assert is_fixed("It opened in 1889.", claim, result, get_checker("lexical"))
    assert not interventions_for(replace(claim, diagnosis=None))
