"""Benchmarks: ``detect`` (RAGTruth) and ``faults`` (triage, verification, attribution).

Run them with ``groundsight bench SUITE [options]``; each suite documents its options
(``groundsight bench SUITE --help``). They need the ``bench`` and ``models`` extras.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence


def run(suite: str, argv: Sequence[str]) -> int:
    module = importlib.import_module(f"groundsight.bench.{suite}")
    code: int = module.main(list(argv))
    return code
