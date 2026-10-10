"""Which passage made the model say each claim?

For every claim, attribution measures how much each passage raises the probability of
the claim's own tokens, by scoring the *actual answer* (teacher forcing) with and
without that passage in the prompt:

- ``loo`` (leave-one-out) removes one passage at a time: ``n + 1`` forward passes.
- ``surrogate`` scores random subsets of passages and fits a linear model of each
  claim's log-probability on which passages were present, the approach popularised by
  ContextCite. It costs a fixed number of passes however many passages there are.

Scores are in nats: a score of 2.3 means the claim was about ten times more likely with
the passage than without it. A small local model can stand in for a hosted one (a
*proxy*); proxy attributions track the target model's well when both read the context
similarly, which the benchmark checks rather than assumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from groundsight.generate import DEFAULT_INSTRUCTION, LocalChat, Message, build_messages
from groundsight.model import Audit, Chunk


class Scorer(Protocol):
    def span_logprobs(
        self, messages: list[Message], answer: str, spans: Sequence[tuple[int, int]]
    ) -> list[float]:
        """Total log-probability of the answer tokens inside each character span."""
        ...


class LocalScorer:
    """Teacher-forced scoring with a local Hugging Face causal LM."""

    def __init__(self, chat: LocalChat | str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        self.chat = chat if isinstance(chat, LocalChat) else LocalChat(chat)
        self.name = self.chat.name

    def span_logprobs(
        self, messages: list[Message], answer: str, spans: Sequence[tuple[int, int]]
    ) -> list[float]:  # pragma: no cover - needs weights; exercised by the benchmarks
        import torch

        prompt = self.chat.prompt_ids(messages)
        tok = self.chat.tokenizer
        encoded = tok(answer, add_special_tokens=False, return_offsets_mapping=True)
        answer_ids = torch.tensor([encoded["input_ids"]], device=prompt.device)
        ids = torch.cat([prompt, answer_ids], dim=1)
        with torch.inference_mode():
            logits = self.chat.model(ids).logits.float()
        start = prompt.shape[1]
        logprobs = torch.log_softmax(logits[0, start - 1 : -1], dim=-1)
        token_lp = logprobs.gather(1, answer_ids[0].unsqueeze(1)).squeeze(1).cpu().numpy()
        offsets = encoded["offset_mapping"]
        out = []
        for s, e in spans:
            mask = [i for i, (a, b) in enumerate(offsets) if a < e and b > s]
            out.append(float(token_lp[mask].sum()) if mask else 0.0)
        return out


@dataclass
class Attribution:
    claim_indices: list[int]
    chunk_ids: list[str]
    scores: np.ndarray  # (claims, chunks): log-probability the passage adds to the claim
    base: np.ndarray  # (claims,): claim log-probability with every passage present
    method: str
    passes: int

    def for_claim(self, claim_index: int) -> list[tuple[str, float]]:
        """Passages ranked by how much they raise this claim's probability."""
        row = self.scores[self.claim_indices.index(claim_index)]
        order = np.argsort(-row, kind="stable")
        return [(self.chunk_ids[j], float(row[j])) for j in order]

    def top(self, claim_index: int) -> str:
        return self.for_claim(claim_index)[0][0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_indices": self.claim_indices,
            "chunk_ids": self.chunk_ids,
            "scores": np.round(self.scores, 4).tolist(),
            "base": np.round(self.base, 4).tolist(),
            "method": self.method,
            "passes": self.passes,
        }


def _ridge(design: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge regression with an unpenalised intercept; returns the slopes only."""
    x = np.column_stack([np.ones(len(design)), design])
    penalty = alpha * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ targets)
    return np.asarray(coef[1:])


def attribute(
    audit: Audit,
    scorer: Scorer,
    *,
    method: Literal["auto", "loo", "surrogate"] = "auto",
    claims: Sequence[int] | None = None,
    ablations: int = 32,
    keep: float = 0.5,
    alpha: float = 0.01,
    instruction: str = DEFAULT_INSTRUCTION,
    seed: int = 0,
) -> Attribution:
    """Attribute each claim of ``audit.answer`` to the passages in its context."""
    chosen = [c for c in audit.claims if claims is None or c.index in claims]
    spans = [(c.start, c.end) for c in chosen]
    chunks = audit.chunks
    n = len(chunks)

    def score(subset: Sequence[Chunk]) -> np.ndarray:
        messages = build_messages(audit.question, subset, instruction)
        return np.asarray(scorer.span_logprobs(messages, audit.answer, spans))

    base = score(chunks)
    if method == "auto":
        method = "loo" if n <= 12 else "surrogate"
    if method == "loo":
        scores = np.zeros((len(chosen), n))
        for j in range(n):
            scores[:, j] = base - score([c for i, c in enumerate(chunks) if i != j])
        passes = n + 1
    else:
        rng = np.random.default_rng(seed)
        masks = rng.random((ablations, n)) < keep
        values = np.stack(
            [score([c for c, m in zip(chunks, row, strict=True) if m]) for row in masks]
        )
        scores = np.stack(
            [_ridge(masks.astype(float), values[:, k], alpha) for k in range(len(chosen))]
        )
        passes = ablations + 1
    return Attribution(
        claim_indices=[c.index for c in chosen],
        chunk_ids=[c.id for c in chunks],
        scores=scores,
        base=base,
        method=method,
        passes=passes,
    )
