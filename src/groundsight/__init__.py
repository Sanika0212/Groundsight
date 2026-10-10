"""Groundsight: see which claims in a RAG answer your sources don't support, and why.

.. code-block:: python

    import groundsight as gs

    result = gs.audit(answer, chunks, question=question)
    for claim in result.flagged:
        print(claim.verdict, claim.text, claim.diagnosis.cause)
    gs.write_report(result, "report.html")

"""

from groundsight._version import __version__
from groundsight.attribution import Attribution, LocalScorer, attribute
from groundsight.audit import DEFAULT_CHECKER, audit
from groundsight.calibrate import Profile, fit_profile
from groundsight.capture import Store, capture_http, current, generator, retriever, trace
from groundsight.checkers import load_checker
from groundsight.generate import (
    AnthropicChat,
    LocalChat,
    OpenAIChat,
    PipelineResponder,
    PromptResponder,
)
from groundsight.model import Audit, Cause, Chunk, ClaimCheck, Thresholds, Verdict
from groundsight.report import render, write_report
from groundsight.triage import diagnose
from groundsight.verify import estimate, verify

__all__ = [
    "DEFAULT_CHECKER",
    "AnthropicChat",
    "Attribution",
    "Audit",
    "Cause",
    "Chunk",
    "ClaimCheck",
    "LocalChat",
    "LocalScorer",
    "OpenAIChat",
    "PipelineResponder",
    "Profile",
    "PromptResponder",
    "Store",
    "Thresholds",
    "Verdict",
    "__version__",
    "attribute",
    "audit",
    "capture_http",
    "current",
    "diagnose",
    "estimate",
    "fit_profile",
    "generator",
    "load_checker",
    "render",
    "retriever",
    "trace",
    "verify",
    "write_report",
]
