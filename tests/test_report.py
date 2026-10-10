import json
import re
from pathlib import Path

import numpy as np

from groundsight.attribution import Attribution
from groundsight.audit import audit
from groundsight.generate import PipelineResponder
from groundsight.model import Thresholds
from groundsight.report import render, write_report
from groundsight.verify import verify

LIMITS = Thresholds(support=0.6, contradiction=0.5, calibration="test <thresholds>")
TOWER = "The Eiffel Tower is 330 metres tall. It was completed in 1889."


def make():  # type: ignore[no-untyped-def]
    return audit(
        "The Eiffel Tower is 500 metres tall. <b>Bold</b> claims need sources here.",
        [TOWER, "Text with </script> inside."],
        question="How tall? <script>alert(1)</script>",
        checker="lexical",
        thresholds=LIMITS,
    )


def test_report_is_self_contained_escaped_and_complete(tmp_path: Path) -> None:
    result = make()
    page = render(result)
    assert page.startswith("<!doctype html>")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "&lt;b&gt;Bold&lt;/b&gt;" in page
    assert "test &lt;thresholds&gt;" in page
    assert not re.search(r"https?://", page)
    assert page.count('class="claim ') == 2
    assert "Misread context" in page
    payload = page.split('id="gs-data">')[1].split("</script>")[0]
    data = json.loads(payload)
    assert data["initial"] == 0
    assert data["chunks"][1]["text"] == "Text with </script> inside."
    assert "All claims as a table" in page
    path = write_report(result, tmp_path / "deep" / "r.html")
    assert path.read_text() == page


def test_report_with_attribution_and_verification() -> None:
    result = make()
    attribution = Attribution(
        claim_indices=[0, 1],
        chunk_ids=["c0", "c1"],
        scores=np.array([[3.0, -0.5], [0.2, 0.1]]),
        base=np.array([-4.0, -9.0]),
        method="loo",
        passes=3,
    )
    responder = PipelineResponder(
        lambda q, chunks, instruction, seed: "The tower is 330 metres tall."
    )
    checks = verify(result, responder, samples=2, claims=[0])
    page = render(result, attribution=attribution, verifications=checks)
    assert "Which passage made the model say each claim" in page
    assert "loo, 3 forward passes" in page
    assert "Effect vs resampling" in page
    assert "move c0 to the top" in page
    assert "(not confirmed)" in page or "(confirmed)" in page


def test_report_without_flags() -> None:
    clean = audit(
        "The Eiffel Tower is 330 metres tall.", [TOWER], checker="lexical", thresholds=LIMITS
    )
    assert "No flagged claims" in render(clean)
