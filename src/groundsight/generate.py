"""Answer generators used to replay a RAG answer under a counterfactual change.

A *responder* turns ``(question, chunks, instruction, seed)`` into an answer. Wrap your
own pipeline's generation step with :class:`PipelineResponder`, or use a built-in
generator with the default RAG prompt via :class:`PromptResponder`:

* :class:`OpenAIChat` for OpenAI or any OpenAI-compatible endpoint,
* :class:`AnthropicChat` for the Anthropic Messages API,
* :class:`LocalChat` for a Hugging Face instruct model run locally.

API keys are read from the environment (or passed explicitly) and never logged.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from groundsight.model import Chunk

Message = dict[str, str]

DEFAULT_INSTRUCTION = (
    "Answer the question using the numbered context passages. Be concise and factual."
)
STRICT_INSTRUCTION = (
    "Answer the question using ONLY the numbered context passages. Do not add facts that "
    "the passages do not state. If the passages do not contain the answer, say that the "
    "context does not say."
)


def build_messages(
    question: str | None, chunks: Sequence[Chunk], instruction: str = DEFAULT_INSTRUCTION
) -> list[Message]:
    context = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question or ''}"},
    ]


class Generator(Protocol):
    name: str

    def __call__(
        self, messages: list[Message], *, temperature: float = 0.0, seed: int | None = None
    ) -> str: ...


class Responder(Protocol):
    def __call__(
        self,
        question: str | None,
        chunks: Sequence[Chunk],
        instruction: str | None,
        seed: int,
    ) -> str: ...


@dataclass
class PromptResponder:
    """Answer with ``generator`` using the default RAG prompt (or a given instruction)."""

    generator: Generator
    temperature: float = 0.7
    instruction: str = DEFAULT_INSTRUCTION

    def __call__(
        self, question: str | None, chunks: Sequence[Chunk], instruction: str | None, seed: int
    ) -> str:
        messages = build_messages(question, chunks, instruction or self.instruction)
        return self.generator(messages, temperature=self.temperature, seed=seed)


@dataclass
class PipelineResponder:
    """Use your own generation step: ``fn(question, chunks, instruction, seed) -> answer``.

    ``instruction`` is ``None`` unless an intervention asks for a stricter prompt; pass it
    into your system prompt to support instruction interventions.
    """

    fn: Callable[[str | None, Sequence[Chunk], str | None, int], str]

    def __call__(
        self, question: str | None, chunks: Sequence[Chunk], instruction: str | None, seed: int
    ) -> str:
        return self.fn(question, chunks, instruction, seed)


class GenerationError(RuntimeError):
    """The model API refused a request (no credit, bad key, unknown model, ...)."""


_RETRY = {429, 500, 502, 503, 504, 529}
_FATAL = {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}


@dataclass
class _HTTPChat:
    model: str
    api_key: str | None = None
    max_tokens: int = 256
    client: httpx.Client | None = None
    max_retries: int = 4
    sleep: Callable[[float], None] = time.sleep
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    name: str = field(default="http", init=False)

    def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> Any:
        http = self.client or httpx.Client(timeout=120.0)
        self.client = http
        for attempt in range(self.max_retries + 1):
            response = http.post(url, headers=headers, json=payload)
            if response.is_success:
                self.calls += 1
                return response.json()
            try:
                error = response.json().get("error", {})
            except ValueError:
                error = {}
            if not isinstance(error, dict):
                error = {"message": str(error)}
            code = str(error.get("code") or error.get("type") or "")
            retry = response.status_code in _RETRY and code not in _FATAL
            if not retry or attempt == self.max_retries:
                message = str(error.get("message") or response.reason_phrase)[:300]
                raise GenerationError(
                    f"{self.name} returned {response.status_code} ({code or 'error'}: {message})"
                )
            wait = response.headers.get("retry-after", "")
            self.sleep(float(wait) if wait.replace(".", "", 1).isdigit() else 2.0**attempt)
        raise AssertionError("unreachable")  # pragma: no cover


@dataclass
class OpenAIChat(_HTTPChat):
    model: str = "gpt-4.1-mini"
    base_url: str | None = None

    def __post_init__(self) -> None:
        self.name = f"openai:{self.model}"
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            self.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        if not self.api_key:
            raise GenerationError("set OPENAI_API_KEY or pass api_key")

    def __call__(
        self, messages: list[Message], *, temperature: float = 0.0, seed: int | None = None
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        body = self._post(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            payload,
        )
        usage = body.get("usage", {})
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        return str(body["choices"][0]["message"].get("content") or "")


@dataclass
class AnthropicChat(_HTTPChat):
    model: str = "claude-haiku-5-5"

    def __post_init__(self) -> None:
        self.name = f"anthropic:{self.model}"
        self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise GenerationError("set ANTHROPIC_API_KEY or pass api_key")

    def __call__(
        self, messages: list[Message], *, temperature: float = 0.0, seed: int | None = None
    ) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        body = self._post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": str(self.api_key), "anthropic-version": "2023-06-01"},
            {
                "model": self.model,
                "system": system,
                "messages": turns,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
            },
        )
        usage = body.get("usage", {})
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        return "".join(b.get("text", "") for b in body.get("content", []))


class LocalChat:
    """A Hugging Face instruct model run locally (sampling is seeded per call)."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        device: str | None = None,
        max_new_tokens: int = 128,
        dtype: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.name = f"local:{model_id}"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.model: Any = None
        self.tokenizer: Any = None
        self.calls = 0

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from groundsight.checkers import default_device

        self.device = self.device or default_device()
        dtype = torch.float16 if self.dtype == "auto" and self.device != "cpu" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype)
        self.model.to(self.device).eval()

    def prompt_ids(self, messages: list[Message]) -> Any:
        """Token ids of the chat prompt, ending where the assistant's reply begins."""
        self.load()
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return encoded["input_ids"].to(self.device)

    def __call__(
        self, messages: list[Message], *, temperature: float = 0.0, seed: int | None = None
    ) -> str:  # pragma: no cover - exercised by the benchmarks, needs weights
        import torch

        ids = self.prompt_ids(messages)
        if seed is not None:
            torch.manual_seed(seed)
        sample = temperature > 0
        with torch.inference_mode():
            out = self.model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=sample,
                temperature=temperature if sample else None,
                top_p=0.95 if sample else None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self.calls += 1
        return str(self.tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)).strip()
