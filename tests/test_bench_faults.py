import json
import random
import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from groundsight.bench import faults
from groundsight.bench.faults import (
    Instance,
    Question,
    build_instances,
    correct,
    edit,
    evaluate_attribution,
    evaluate_triage,
    evaluate_verification,
    markdown,
    normalize,
    substitute,
    summarize,
)


class Reader:
    """Fake answering model: reads only the first eight passages, trusts what it sees."""

    name = "reader"

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers  # question -> right answer

    def __call__(self, messages, *, temperature: float = 0.0, seed=None) -> str:  # type: ignore[no-untyped-def]
        user = messages[1]["content"]
        question = user.rsplit("Question: ", 1)[1]
        passages = re.findall(r"\[(\d+)\] (.*)", user)
        seen = " ".join(text for number, text in passages if int(number) <= 8)
        right = self.answers[question]
        match = re.search(r"born in (\w+)", seen)
        if right in seen and match and match.group(1) == right:
            return f"The answer is {right}."
        if match:
            return f"The answer is {match.group(1)}."
        return "The context does not say."


def questions(n: int = 6) -> list[Question]:
    cities = ["Lyon", "Turin", "Porto", "Ghent", "Basel", "Leeds", "Cork", "Graz"]
    out = []
    for i in range(n):
        city = cities[i]
        gold = [f"Person{i}: Person{i} was born in {city} in 1900.", f"Place{i}: {city} is a city."]
        distractors = [f"Other{i}-{j}: An unrelated note number {j}." for j in range(8)]
        out.append(Question(f"q{i}", f"Where was Person{i} born?", city, gold, distractors))
    return out


def test_text_helpers() -> None:
    assert normalize("The  Paris!") == "paris"
    assert correct("It was the city of Paris.", "paris")
    assert edit("Born in LYON.", "Lyon", "Rome") == "Born in Rome."
    rng = random.Random(0)
    assert substitute("1,200", [], rng, "") is not None
    assert substitute("Lyon", ["Rome", "New York", "lyon"], rng, "context") == "Rome"
    assert substitute("Lyon", ["Rome"], rng, "Rome is here") is None
    assert substitute("1.2.3", [], rng, "") is None


def test_pipeline_on_a_fake_reader(tmp_path: Path) -> None:
    qs = questions()
    reader = Reader({q.question: q.answer for q in qs})
    lines: list[str] = []
    instances, stats = build_instances(qs, reader, long_context=12, progress=lines.append)
    assert stats["clean_correct"] >= 3
    kinds = {i.fault for i in instances}
    assert {"retrieval_miss", "misleading_source"} <= kinds
    assert all(not correct(i.faulted_answer, i.answer) for i in instances)
    triage, audits = evaluate_triage(instances, "lexical")
    assert len(triage) == len(instances)
    cache = tmp_path / "verification.jsonl"
    verification = evaluate_verification(
        instances, audits, reader, samples=2, progress=lines.append, cache=cache
    )
    assert len(verification) == len(instances)

    class Unused:
        def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("cached rows must not call the model again")

    flagged = [r for r in verification if r["verified"] is not None]
    if flagged:
        assert cache.exists()
        again = evaluate_verification(instances, audits, Unused(), samples=2, cache=cache)  # type: ignore[arg-type]
        assert again == verification

    class Scorer:
        def span_logprobs(self, messages, answer: str, spans: Sequence[tuple[int, int]]):  # type: ignore[no-untyped-def]
            word = answer.rstrip(".").split()[-1]
            hits = [p for p in messages[1]["content"].split("\n\n") if word in p]
            return [float(len(hits)) for _ in spans]

    attribution = evaluate_attribution(instances, audits, {"fake": Scorer()})
    assert set(attribution) >= {"fake", "word overlap", "random"}
    summary = summarize(triage, verification, attribution, stats)
    assert summary["instances"] == len(instances)
    table = markdown(summary)
    assert "Injected fault" in table
    assert "Attribution method" in table


def test_main_uses_cached_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    qs = questions(4)
    reader = Reader({q.question: q.answer for q in qs})
    instances, stats = build_instances(qs, reader, long_context=12, progress=lambda s: None)
    out = tmp_path / "faults"
    out.mkdir()
    (out / "instances.json").write_text(
        json.dumps({"instances": [i.__dict__ for i in instances], "stats": stats})
    )

    class FakeChat:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.name = "fake"

        def __call__(self, messages, *, temperature=0.0, seed=None):  # type: ignore[no-untyped-def]
            return reader(messages)

    class FakeScorer:
        def __init__(self, *args) -> None:  # type: ignore[no-untyped-def]
            pass

        def span_logprobs(self, messages, answer, spans):  # type: ignore[no-untyped-def]
            return [0.0 for _ in spans]

    class FakeCosine:
        def score(self, pairs):  # type: ignore[no-untyped-def]
            return [(0.5, 0.0) for _ in pairs]

    monkeypatch.setattr("groundsight.generate.LocalChat", FakeChat)
    monkeypatch.setattr("groundsight.generate.OpenAIChat", FakeChat)
    monkeypatch.setattr("groundsight.attribution.LocalScorer", FakeScorer)
    monkeypatch.setattr("groundsight.checkers.CosineChecker", FakeCosine)
    assert faults.main(["--out", str(out), "--checker", "lexical", "--samples", "1"]) == 0
    assert "Triage correct" in (out / "results.md").read_text()
    assert isinstance(instances[0], Instance)


def test_remote_models_ask_first_and_estimate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert faults.estimate_calls(100, 3) > faults.estimate_calls(100, 0)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    generator, remote = faults.make_generator("openai:gpt-x")
    assert remote
    assert generator.name == "openai:gpt-x"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert faults.make_generator("anthropic:m")[1]
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert faults.main(["--model", "openai:gpt-x", "--out", str(tmp_path)]) == 1
