"""Claim checkers: does a premise (a source passage) support or contradict a claim?

A checker scores ``(premise, claim)`` pairs and returns ``(support, contradiction)``
probabilities for each. Three-way checkers (NLI) estimate both; binary fact-checkers
(MiniCheck, HHEM) estimate support only and report a contradiction of 0, so they are
usually combined with an NLI model that supplies the contradiction score.

Available through :func:`load_checker`:

=================  =========================================================  ==========
name               model                                                      output
=================  =========================================================  ==========
``lexical``        word overlap with number and negation checks (no model)    3-way
``cosine``         embedding similarity, all-MiniLM-L6-v2 (a baseline only)   support
``nli``            cross-encoder/nli-deberta-v3-base                          3-way
``nli-large``      cross-encoder/nli-deberta-v3-large                         3-way
``minicheck``      lytang/MiniCheck-RoBERTa-Large                             support
``hhem``           vectara/hallucination_evaluation_model (remote code)       support
``A+B``            support from ``A``, contradiction from ``B``               3-way
=================  =========================================================  ==========

Model weights are downloaded on first use and run locally; no text leaves the machine.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    import torch

Pair = tuple[str, str]
Score = tuple[float, float]


class Checker(Protocol):
    name: str
    three_way: bool

    def score(self, pairs: Sequence[Pair]) -> list[Score]:
        """``(support, contradiction)`` for each ``(premise, claim)`` pair."""
        ...


# --------------------------------------------------------------------------- lexical

STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there here of in on at to for
    from by with without about into onto over under as is are was were be been being am
    it its it's they them their he him his she her we us our you your i me my do does did
    done has have had having will would can could should may might must shall so such
    also very just only which who whom whose what when where why how all any each both
    some most more other another same own not no nor never none
    """.split()
)
NEGATIONS = frozenset({"not", "no", "never", "none", "nor", "cannot", "without", "n't"})
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*|n't")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("’", "'"))


def content_words(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS and len(t) > 1}


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def overlap(premise: str, claim: str) -> float:
    """Share of the claim's content words that appear in the premise (topical relevance)."""
    claim_words = content_words(claim)
    if not claim_words:
        return 0.0
    return len(claim_words & content_words(premise)) / len(claim_words)


class LexicalChecker:
    """Word-overlap checker with number and negation conflict rules.

    It needs no model and is deterministic, which makes it useful in tests, in CI and
    as a floor in benchmarks. It cannot recognise paraphrase, so prefer a model-based
    checker for real audits.
    """

    name = "lexical"
    three_way = True

    def score(self, pairs: Sequence[Pair]) -> list[Score]:
        return [self._one(premise, claim) for premise, claim in pairs]

    @staticmethod
    def _one(premise: str, claim: str) -> Score:
        cover = overlap(premise, claim)
        claim_numbers, premise_numbers = numbers(claim), numbers(premise)
        missing_numbers = claim_numbers - premise_numbers
        claim_neg = bool(NEGATIONS & set(tokens(claim)))
        premise_neg = bool(NEGATIONS & set(tokens(premise)))
        support = cover
        contradiction = 0.0
        if missing_numbers:
            support *= 0.3
            if premise_numbers and cover >= 0.5:
                contradiction = 0.8
        if claim_neg != premise_neg and cover >= 0.6:
            support *= 0.3
            contradiction = max(contradiction, 0.7)
        return float(support), float(contradiction)


# ----------------------------------------------------------------- transformer models


def default_device() -> str:
    """``GROUNDSIGHT_DEVICE`` if set, else CUDA, else Apple MPS, else CPU."""
    override = os.environ.get("GROUNDSIGHT_DEVICE")
    if override:
        return override
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum(axis=-1, keepdims=True))


class _PairClassifier:
    """A Hugging Face sequence-pair classifier scored in batches, with a result cache."""

    name = "pair-classifier"
    three_way = False

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self._model: Any = None
        self._tokenizer: Any = None
        self._cache: dict[Pair, Score] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = self.device or default_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        self._model.to(self.device).eval()

    def _forward(self, batch: Sequence[Pair]) -> np.ndarray:
        """Logits of shape ``(len(batch), num_labels)``."""
        import torch

        self._load()
        encoded = self._tokenizer(
            [p for p, _ in batch],
            [c for _, c in batch],
            truncation="only_first",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits: torch.Tensor = self._model(**encoded).logits
        return np.asarray(logits.float().cpu().numpy())

    def _convert(self, logits: np.ndarray) -> list[Score]:
        probs = _softmax(logits)
        return [(float(p[-1]), 0.0) for p in probs]

    def score(self, pairs: Sequence[Pair]) -> list[Score]:
        todo = list(dict.fromkeys(p for p in pairs if p not in self._cache))
        # Similar lengths in a batch waste less padding.
        todo.sort(key=lambda p: len(p[0]) + len(p[1]))
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i : i + self.batch_size]
            for pair, result in zip(batch, self._convert(self._forward(batch)), strict=True):
                self._cache[pair] = result
        return [self._cache[p] for p in pairs]


class NLIChecker(_PairClassifier):
    """Three-way natural language inference cross-encoder."""

    three_way = True

    def __init__(self, model_id: str = "cross-encoder/nli-deberta-v3-base", **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        self.name = "nli" if model_id.endswith("base") else f"nli:{model_id}"
        self._labels: tuple[int, int] | None = None

    def label_indices(self, id2label: dict[int, str]) -> tuple[int, int]:
        """Indices of the entailment and contradiction labels in the model's output."""
        names = {int(i): str(label).lower() for i, label in id2label.items()}
        entail = next(i for i, n in names.items() if n.startswith("entail"))
        contra = next(i for i, n in names.items() if n.startswith("contradict"))
        return entail, contra

    def _convert(self, logits: np.ndarray) -> list[Score]:
        if self._labels is None:
            self._load()
            self._labels = self.label_indices(self._model.config.id2label)
        entail, contra = self._labels
        probs = _softmax(logits)
        return [(float(p[entail]), float(p[contra])) for p in probs]


class MiniCheckChecker(_PairClassifier):
    """MiniCheck fact-checker: probability that the document supports the claim."""

    def __init__(self, model_id: str = "lytang/MiniCheck-RoBERTa-Large", **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        self.name = "minicheck"


class HHEMChecker(_PairClassifier):
    """Vectara's HHEM-2.1-open consistency model (runs the model's own remote code)."""

    def __init__(
        self, model_id: str = "vectara/hallucination_evaluation_model", **kwargs: Any
    ) -> None:
        kwargs.setdefault("trust_remote_code", True)
        super().__init__(model_id, **kwargs)
        self.name = "hhem"

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification

        self.device = self.device or default_device()
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model.to(self.device).eval()

    def _forward(self, batch: Sequence[Pair]) -> np.ndarray:
        import torch

        self._load()
        with torch.inference_mode():
            scores = self._model.predict(list(batch))
        return np.asarray(scores.float().cpu().numpy()).reshape(-1, 1)

    def _convert(self, logits: np.ndarray) -> list[Score]:
        return [(float(s[0]), 0.0) for s in logits]  # already a probability


class CosineChecker(_PairClassifier):
    """Embedding cosine similarity, the usual "groundedness" shortcut.

    Similarity measures whether two texts are about the same thing, not whether one
    supports the other, so a claim that contradicts its source still scores high. It
    is kept as a baseline for the benchmark and as a relevance signal for triage.
    """

    def __init__(
        self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs: Any
    ) -> None:
        super().__init__(model_id, **kwargs)
        self.name = "cosine"

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self.device = self.device or default_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self._load()
        encoded = self._tokenizer(
            list(texts), truncation=True, max_length=256, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            hidden = self._model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return np.asarray(pooled.float().cpu().numpy())

    def _forward(self, batch: Sequence[Pair]) -> np.ndarray:
        premises = self.embed([p for p, _ in batch])
        claims = self.embed([c for _, c in batch])
        return np.asarray((premises * claims).sum(-1)).reshape(-1, 1)

    def _convert(self, logits: np.ndarray) -> list[Score]:
        return [(float(max(0.0, min(1.0, s[0]))), 0.0) for s in logits]


class Combined:
    """Support from one checker, contradiction from another (usually an NLI model)."""

    three_way = True

    def __init__(self, support: Checker, contradiction: Checker) -> None:
        self.support_checker = support
        self.contradiction_checker = contradiction
        self.name = f"{support.name}+{contradiction.name}"

    def score(self, pairs: Sequence[Pair]) -> list[Score]:
        supports = self.support_checker.score(pairs)
        contradictions = self.contradiction_checker.score(pairs)
        return [(s, c) for (s, _), (_, c) in zip(supports, contradictions, strict=True)]


def load_checker(spec: str = "minicheck+nli", **kwargs: Any) -> Checker:
    """Build a checker from a name such as ``"nli"`` or ``"minicheck+nli"``."""
    if "+" in spec:
        support, contradiction = spec.split("+", 1)
        return Combined(load_checker(support, **kwargs), load_checker(contradiction, **kwargs))
    factories: dict[str, Any] = {
        "lexical": lambda: LexicalChecker(),
        "cosine": lambda: CosineChecker(**kwargs),
        "nli": lambda: NLIChecker(**kwargs),
        "nli-large": lambda: NLIChecker("cross-encoder/nli-deberta-v3-large", **kwargs),
        "minicheck": lambda: MiniCheckChecker(**kwargs),
        "hhem": lambda: HHEMChecker(**kwargs),
    }
    if spec not in factories:
        raise ValueError(f"unknown checker {spec!r}; choose from {sorted(factories)} or A+B")
    checker: Checker = factories[spec]()
    return checker
