"""Capture what a RAG pipeline retrieved and answered, then audit it.

Mark the retrieval and generation steps of your pipeline, or record them by hand:

.. code-block:: python

    import groundsight as gs

    @gs.retriever
    def search(query): ...          # returns strings, dicts or LangChain-style documents

    @gs.generator
    def answer(query, docs): ...    # returns the answer text

    with gs.trace("What is our refund window?") as t:
        answer(t.question, search(t.question))
    print(t.audit.groundedness)

Inside a trace, :func:`capture_http` can also record OpenAI- and Anthropic-style chat
calls made through ``httpx`` (the transport both official SDKs use), so the generation
step needs no decorator. Audits are appended to a local :class:`Store`.
"""

from __future__ import annotations

import functools
import json
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from groundsight.audit import ChunkLike, as_chunks
from groundsight.audit import audit as run_audit
from groundsight.model import Audit, Chunk

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_STORE = Path(".groundsight") / "audits.jsonl"


class Store:
    """Append-only JSON Lines file of audits (the newest record of an id wins)."""

    def __init__(self, path: str | Path = DEFAULT_STORE) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def save(self, result: Audit) -> Audit:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(result.to_dict(), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return result

    def _records(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return records
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                data = json.loads(line)
                records.pop(data["id"], None)  # re-insert so order follows the latest save
                records[data["id"]] = data
        return records

    def all(self) -> list[Audit]:
        """Audits, most recent first."""
        return [Audit.from_dict(d) for d in reversed(list(self._records().values()))]

    def get(self, audit_id: str) -> Audit:
        records = self._records()
        matches = [k for k in records if k.startswith(audit_id)]
        if len(matches) != 1:
            raise KeyError(f"{audit_id!r} matches {len(matches)} audits")
        return Audit.from_dict(records[matches[0]])


@dataclass
class Trace:
    question: str | None = None
    chunks: list[Chunk] = field(default_factory=list)
    answer: str | None = None
    model: str | None = None
    messages: list[dict[str, Any]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    audit: Audit | None = None

    def retrieved(self, chunks: Sequence[ChunkLike | Any]) -> list[Chunk]:
        """Record retrieved passages (appended, so multi-step retrieval accumulates)."""
        offset = len(self.chunks)
        new = as_chunks([_plain(c) for c in chunks])
        renamed = [
            Chunk(f"c{offset + i}", c.text, c.score, c.source, c.metadata) if c.id == f"c{i}" else c
            for i, c in enumerate(new)
        ]
        self.chunks.extend(renamed)
        return renamed

    def answered(self, answer: str, *, model: str | None = None) -> None:
        self.answer = answer
        self.model = model or self.model


def _plain(item: Any) -> ChunkLike:
    """Turn LangChain/LlamaIndex-style documents into dicts."""
    if isinstance(item, (str, Chunk, dict)):
        return item
    text = getattr(item, "page_content", None) or getattr(item, "text", None)
    if text is None:
        return str(item)
    meta = dict(getattr(item, "metadata", None) or {})
    doc_id = getattr(item, "id", None) or meta.pop("id", None)
    return {"text": text, **({"id": str(doc_id)} if doc_id else {}), **meta}


_current: ContextVar[Trace | None] = ContextVar("groundsight_trace", default=None)


def current() -> Trace | None:
    return _current.get()


@contextmanager
def trace(
    question: str | None = None,
    *,
    store: Store | str | Path | None = DEFAULT_STORE,
    audit: bool = True,
    **audit_options: Any,
) -> Iterator[Trace]:
    """Record one question's retrieval and answer; audit and store it on exit."""
    record = Trace(question=question)
    token = _current.set(record)
    try:
        yield record
    finally:
        _current.reset(token)
    if audit and record.answer is not None:
        meta = {"model": record.model, **record.meta} if record.model else dict(record.meta)
        record.audit = run_audit(
            record.answer, record.chunks, question=record.question, meta=meta, **audit_options
        )
        if store is not None:
            (store if isinstance(store, Store) else Store(store)).save(record.audit)


def retriever(fn: Callable[P, R]) -> Callable[P, R]:
    """Record the passages ``fn`` returns in the active trace."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        result = fn(*args, **kwargs)
        active = current()
        if active is not None and isinstance(result, (list, tuple)):
            active.retrieved(result)
        return result

    return wrapper


def generator(fn: Callable[P, R]) -> Callable[P, R]:
    """Record the answer ``fn`` returns (a string) in the active trace."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        result = fn(*args, **kwargs)
        active = current()
        if active is not None and isinstance(result, str):
            active.answered(result)
        return result

    return wrapper


# ------------------------------------------------------------------ HTTP capture

_patched: dict[str, tuple[Any, Any, Any]] = {}
_patch_lock = threading.Lock()


def _record_exchange(request: Any, response: Any) -> None:
    active = current()
    path = str(request.url.path)
    if active is None or not (path.endswith("/chat/completions") or path.endswith("/messages")):
        return
    try:
        body = json.loads(request.content or b"{}")
        reply = json.loads(response.content)
    except (ValueError, AttributeError, RuntimeError):
        return  # streaming or non-JSON bodies are left to the decorators
    if "choices" in reply:
        text = reply["choices"][0].get("message", {}).get("content") or ""
    elif "content" in reply:
        text = "".join(b.get("text", "") for b in reply["content"] if isinstance(b, dict))
    else:
        return
    active.messages = body.get("messages")
    active.answered(str(text), model=body.get("model"))


def capture_http() -> list[str]:
    """Record chat completions sent through ``httpx``/``httpx2`` inside a trace.

    Returns the names of the libraries that were patched. Non-streaming responses
    only; use :func:`generator` for streamed answers.
    """
    import importlib

    with _patch_lock:
        for name in ("httpx", "httpx2"):
            if name in _patched:
                continue
            try:
                lib = importlib.import_module(name)
            except ImportError:
                continue
            sync_send, async_send = lib.Client.send, lib.AsyncClient.send

            def send(self: Any, request: Any, *a: Any, _orig: Any = sync_send, **k: Any) -> Any:
                response = _orig(self, request, *a, **k)
                if current() is not None and not k.get("stream"):
                    response.read()
                    _record_exchange(request, response)
                return response

            async def asend(
                self: Any, request: Any, *a: Any, _orig: Any = async_send, **k: Any
            ) -> Any:
                response = await _orig(self, request, *a, **k)
                if current() is not None and not k.get("stream"):
                    await response.aread()
                    _record_exchange(request, response)
                return response

            lib.Client.send, lib.AsyncClient.send = send, asend
            _patched[name] = (lib, sync_send, async_send)
        return sorted(_patched)


def release_http() -> None:
    with _patch_lock:
        for lib, sync_send, async_send in _patched.values():
            lib.Client.send, lib.AsyncClient.send = sync_send, async_send
        _patched.clear()
