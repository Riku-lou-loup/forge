"""Create model clients from an explicit backend selection"""

from forge.agents.model_client import ModelClient
from forge.agents.ollama_client import OllamaClient


def create_client(
    backend: str,
    *,
    model: str = "qwen3.5:4b",
    port: int | None = None,
) -> ModelClient:
    """Create a client without starting servers or making requests"""
    default_ports = {
        "ollama": 11434,  # Local Ollama server
        "ensicompute": 11435,  # SSH tunnel to ensicompute (ensicompute is a GPU server for AI processing tasks for ENSIMAG students)
    }

    if backend not in default_ports:
        raise ValueError(f"Unsupported backend: {backend}")

    selected_port = default_ports[backend] if port is None else port

    return OllamaClient(model=model, port=selected_port)
