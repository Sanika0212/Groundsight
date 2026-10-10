import json
from pathlib import Path

import pandas as pd
import pytest

from groundsight.bench import detect
from groundsight.bench.detect import (
    Example,
    evaluate_contradiction,
    evaluate_detection,
    examples,
    gated,
    passages,
    response_level,
    score,
)


def frame() -> pd.DataFrame:
    rows = []
    for i in range(6):
        bad = i % 2 == 1
        answer = "The fee is 30 dollars. Delivery takes two days."
        if bad:
            answer = "The fee is 90 dollars. Delivery takes two days."
        labels = [{"start": 0, "end": 22, "label_type": "Evident Conflict"}] if bad else []
        for task in detect.TASKS:
            rows.append(
                {
                    "id": f"{task}-{i}",
                    "task_type": task,
                    "query": "What is the fee?",
                    "context": "The fee is 30 dollars.\n\nDelivery takes two days.",
                    "output": answer,
                    "hallucination_labels": json.dumps(labels),
                }
            )
    return pd.DataFrame(rows)


def test_passages_and_examples() -> None:
    assert passages("a\n\n b \n\n", "QA") == ["a", "b"]
    assert passages("  whole doc ", "Summary") == ["whole doc"]
    items = examples(frame(), per_task=4, seed=1)
    assert len(items) == 12
    bad = [e for e in items if e.hallucinated]
    assert bad[0].spans == [(0, 22, "conflict")]
    assert all(len(e.chunks) == (2 if e.task == "QA" else 1) for e in items)
    assert len(examples(frame(), per_task=None)) == 18


def test_scoring_and_metrics() -> None:
    items = examples(frame(), per_task=None)
    rows = score(items, "lexical", progress=False)
    assert len(rows) == 36
    first_bad = next(r for r in rows if r.label)
    assert first_bad.label_type == "conflict"
    assert gated(first_bad, 0.0) >= gated(first_bad, 0.9)
    result = evaluate_detection(rows, rows, "lexical")
    assert result["auroc"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert [c["target"] for c in result["conformal"]] == list(detect.TARGETS)
    assert set(result["by_task"]) == set(detect.TASKS)
    assert response_level(rows, lambda r: r.support < result["threshold"])["f1"] == 1.0
    gates = evaluate_contradiction(rows, gates=(0.0, 0.9))
    assert gates[0]["recall"] == 1.0
    assert gates[1]["recall"] == 0.0


def test_main_runs_end_to_end_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect, "load_split", lambda split: frame())
    out = tmp_path / "detect"
    code = detect.main(
        [
            "--checkers",
            "lexical",
            "--train-per-task",
            "4",
            "--out",
            str(out),
            "--profiles",
            str(tmp_path / "profiles"),
        ]
    )
    assert code == 0
    assert "| lexical |" in (out / "results.md").read_text()
    assert (out / "rows-lexical-test.jsonl").exists()
    assert (tmp_path / "profiles" / "lexical.json").exists()
    # A second run reuses the cached sentence scores.
    assert detect.main(["--checkers", "lexical", "--out", str(out)]) == 0


def test_cache_dir_and_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDSIGHT_CACHE", str(tmp_path))
    frame().to_parquet(tmp_path / "ragtruth-test.parquet")
    assert detect.cache_dir() == tmp_path
    assert len(detect.load_split("test")) == 18
    assert isinstance(Example("i", "QA", "q", [], "a"), Example)
