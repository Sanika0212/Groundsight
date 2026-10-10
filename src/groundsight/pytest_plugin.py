"""Pytest plugin: a ``grounded`` fixture and command-line options.

- ``--groundsight-checker NAME`` picks the checker (default ``GROUNDSIGHT_CHECKER`` or
  the library default).
- ``--groundsight-reports DIR`` writes an HTML report for every failed grounding check.

.. code-block:: python

    def test_answer_is_grounded(grounded):
        grounded(answer, docs, question="What is the refund window?")
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:  # imported lazily so the plugin costs nothing in unrelated test runs
    from groundsight.model import Audit


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("groundsight")
    group.addoption("--groundsight-checker", default=None, help="claim checker to use")
    group.addoption(
        "--groundsight-reports", default=None, help="write HTML reports of failed checks here"
    )


@pytest.fixture
def grounded(request: pytest.FixtureRequest) -> Callable[..., Audit]:
    from groundsight.testing import assert_grounded, default_checker

    checker = request.config.getoption("--groundsight-checker") or default_checker()
    reports = request.config.getoption("--groundsight-reports")
    calls = {"n": 0}

    def check(answer: str, chunks: Any, **kwargs: Any) -> Audit:
        calls["n"] += 1
        if reports and "report" not in kwargs:
            name = request.node.name.replace("/", "_").replace("[", "-").replace("]", "")
            kwargs["report"] = Path(reports) / f"{name}-{calls['n']}.html"
        kwargs.setdefault("checker", checker)
        return assert_grounded(answer, chunks, **kwargs)

    return check
