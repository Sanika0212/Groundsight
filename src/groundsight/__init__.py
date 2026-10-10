"""Groundsight: see which claims in a RAG answer your sources don't support, and why."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("groundsight")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
