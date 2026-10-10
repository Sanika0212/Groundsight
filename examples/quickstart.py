"""Audit a RAG answer, see why its bad claims went wrong, and write a report.

uv run --extra models python examples/quickstart.py            # real checkers
uv run python examples/quickstart.py --checker lexical         # no model download
"""

from __future__ import annotations

import argparse

import groundsight as gs

QUESTION = "How long does the Pro plan keep snapshots, and can I get phone support?"
CHUNKS = [
    "Acme Backup keeps daily snapshots for 30 days on the Standard plan.",
    "The Pro plan keeps daily snapshots for 90 days and adds hourly snapshots for the last "
    "48 hours.",
    "Restores can be started from the web console or with the acme command-line tool.",
    "Deleted files stay recoverable for 30 days after deletion on every plan.",
    "Snapshots are encrypted with AES-256 before they leave your device.",
    "Support is available by email around the clock. Phone support is included with Pro.",
]
ANSWER = (
    "The Pro plan keeps daily snapshots for 60 days, plus hourly snapshots for the last "
    "48 hours. Phone support is included with the Pro plan. Snapshots are stored in three "
    "separate data centres."
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checker", default=gs.DEFAULT_CHECKER)
    args = parser.parse_args()

    result = gs.audit(ANSWER, CHUNKS, question=QUESTION, checker=args.checker)
    print(f"{result.groundedness:.0%} of claims supported")
    for claim in result.claims:
        print(f"[{claim.verdict.value}] {claim.text}")
        if claim.diagnosis:
            print(f"    -> {claim.diagnosis.cause.value}: {claim.diagnosis.reason}")
    gs.write_report(result, "quickstart-report.html")
    print("wrote quickstart-report.html")
