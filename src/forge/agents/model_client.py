"""Common interface for language-model clients used by FORGE"""

from typing import Any, Protocol


class ModelClient(Protocol):
    """Capabilities the investigation agent needs from a model client"""

    def identity(self) -> dict[str, Any]:
        """Return a dictionary describing the model client"""
        ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        timeout: float = 180,
    ) -> dict[str, Any]:
        """Return a normalized assistant message and response metadata"""
        ...


class ModelError(RuntimeError):
    """A model provider could not complete a request"""
