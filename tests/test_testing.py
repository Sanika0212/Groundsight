from pathlib import Path

import pytest

from groundsight.model import Thresholds
from groundsight.testing import GroundingError, assert_grounded, default_checker

DOCS = ["Refunds are accepted within 30 days of delivery.", "Shipping is free over $50."]
LIMITS = Thresholds(support=0.6)


def test_passes_and_fails_with_explanations(tmp_path: Path) -> None:
    ok = assert_grounded(
        "Refunds are accepted within 30 days of delivery.",
        DOCS,
        checker="lexical",
        thresholds=LIMITS,
    )
    assert ok.groundedness == 1.0
    with pytest.raises(GroundingError) as info:
        assert_grounded(
            "Refunds are accepted within 90 days of delivery. Gift cards never expire anywhere.",
            DOCS,
            checker="lexical",
            thresholds=LIMITS,
            report=tmp_path / "fail.html",
        )
    text = str(info.value)
    assert "1 contradicted (allowed 0)" in text
    assert "1 unsupported (allowed 0)" in text
    assert "evidence (c0): Refunds are accepted within 30 days of delivery." in text
    assert (tmp_path / "fail.html").exists()
    assert info.value.audit.groundedness == 0.0
    relaxed = assert_grounded(
        "Gift cards never expire anywhere.", DOCS, checker="lexical", max_unsupported=1
    )
    assert relaxed.counts()["unsupported"] == 1
    with pytest.raises(GroundingError, match="groundedness 0% < 50%"):
        assert_grounded(
            "Gift cards never expire anywhere.",
            DOCS,
            checker="lexical",
            max_unsupported=5,
            min_groundedness=0.5,
        )
    with pytest.raises(GroundingError, match="contradicts the reference"):
        assert_grounded(
            "Shipping is free over $50.",
            ["Shipping is free over $50."],
            reference="Shipping is free over $80.",
            checker="lexical",
            thresholds=LIMITS,
        )


def test_default_checker_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDSIGHT_CHECKER", "lexical")
    assert default_checker() == "lexical"


def test_pytest_fixture_and_reports(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_rag="""
DOCS = ["Refunds are accepted within 30 days of delivery."]

def test_good(grounded):
    grounded("Refunds are accepted within 30 days of delivery.", DOCS)

def test_bad(grounded):
    grounded("Refunds are accepted within 90 days of delivery.", DOCS)
"""
    )
    result = pytester.runpytest("--groundsight-checker", "lexical", "--groundsight-reports", "out")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*answer is not grounded*contradicted*"])
    assert (pytester.path / "out" / "test_bad-1.html").exists()
