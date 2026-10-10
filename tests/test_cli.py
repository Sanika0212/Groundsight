import json
from pathlib import Path

import pytest

from groundsight.calibrate import Profile
from groundsight.cli import DEMO, main, make_responder
from groundsight.generate import PromptResponder


def test_audit_report_export_and_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    store = str(tmp_path / "audits.jsonl")
    case = tmp_path / "case.json"
    case.write_text(json.dumps(DEMO))
    assert (
        main(
            [
                "audit",
                str(case),
                "--checker",
                "lexical",
                "--store",
                store,
                "--html",
                str(tmp_path / "r.html"),
                "--json",
                str(tmp_path / "a.json"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "claims supported" in out
    audit_id = json.loads((tmp_path / "a.json").read_text())["id"]
    assert (tmp_path / "r.html").exists()
    lines = tmp_path / "many.jsonl"
    lines.write_text("\n".join(json.dumps(DEMO) for _ in range(2)) + "\n")
    code = main(
        [
            "audit",
            str(lines),
            "--checker",
            "lexical",
            "--html",
            str(tmp_path / "m.html"),
            "--fail-on-flags",
        ]
    )
    assert code == 1
    assert (tmp_path / "m-1.html").exists()
    listing = tmp_path / "list.json"
    listing.write_text(json.dumps([DEMO]))
    assert main(["audit", str(listing), "--checker", "lexical", "--no-diagnose"]) == 0
    capsys.readouterr()

    assert main(["report", audit_id[:8], "--store", store, "--out", str(tmp_path / "x.html")]) == 0
    assert main(["export", audit_id, "--store", store, "--out", str(tmp_path / "o.json")]) == 0
    assert json.loads((tmp_path / "o.json").read_text())["resourceSpans"]

    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert main(["verify", audit_id, "--store", store, "--model", "openai:gpt-x"]) == 1
    assert "cancelled" in capsys.readouterr().out
    assert main(["version"]) == 0


def test_verify_with_local_style_responder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    store = str(tmp_path / "audits.jsonl")
    case = tmp_path / "case.json"
    case.write_text(json.dumps(DEMO))
    main(
        [
            "audit",
            str(case),
            "--checker",
            "lexical",
            "--store",
            store,
            "--json",
            str(tmp_path / "a.json"),
        ]
    )
    audit_id = json.loads((tmp_path / "a.json").read_text())["id"]

    class Fake:
        name = "fake"

        def __call__(self, messages, *, temperature=0.0, seed=None):  # type: ignore[no-untyped-def]
            return "The Pro plan keeps daily snapshots for 90 days."

    monkeypatch.setattr(
        "groundsight.cli.make_responder", lambda spec, temperature: PromptResponder(Fake())
    )
    html = tmp_path / "v.html"
    assert (
        main(
            [
                "verify",
                audit_id,
                "--store",
                store,
                "--model",
                "local:fake",
                "--samples",
                "2",
                "--html",
                str(html),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "verification needs" in out
    assert "Effect vs resampling" in html.read_text()


def test_make_responder_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert make_responder("openai:gpt-x", 0.5).generator.name == "openai:gpt-x"
    assert make_responder("anthropic:m", 0.5).generator.name == "anthropic:m"
    assert make_responder("anthropic", 0.5).generator.name.startswith("anthropic:")
    assert make_responder("local:tiny/model", 0.5).generator.name == "local:tiny/model"
    with pytest.raises(SystemExit, match="unknown model"):
        make_responder("cohere:x", 0.5)


def test_calibrate_and_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "labels.jsonl"
    rows = []
    for i in range(30):
        rows.append(
            {
                "claim": f"The fee is {i} dollars.",
                "chunks": [f"The fee is {i} dollars."],
                "hallucinated": False,
            }
        )
        rows.append(
            {
                "claim": f"The tax is {i + 100} euros today.",
                "chunks": [f"The fee is {i} dollars."],
                "hallucinated": True,
            }
        )
    data.write_text("\n".join(json.dumps(r) for r in rows))
    out_profile = tmp_path / "p.json"
    assert main(["calibrate", str(data), "--checker", "lexical", "--out", str(out_profile)]) == 0
    printed = capsys.readouterr().out
    assert "60 claims, AUROC 1.000" in printed
    assert "recall >= 90%" in printed
    assert Profile.load(out_profile).checker == "lexical"
    assert main(["demo", "--checker", "lexical", "--out", str(tmp_path / "d.html")]) == 0
    assert (tmp_path / "d.html").exists()


def test_package_exports_are_the_functions_not_the_submodules() -> None:
    import groundsight
    import groundsight.audit
    import groundsight.verify

    assert callable(groundsight.audit)
    assert callable(groundsight.verify)
    assert set(groundsight.__all__) <= set(dir(groundsight))
