"""Small local-only Ollama HTTP adapter. No SDK, proxy, redirects or cloud fallback."""

import http.client
import json
import re

from forge.agents.model_client import ModelError


class OllamaError(ModelError):
    """A local model request could not be completed safely."""


class OllamaClient:
    def __init__(self, model="qwen3.5:4b", *, port=11434, context=4096):
        if (
            not isinstance(model, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,100}", model)
            or "cloud" in model.lower()
        ):
            raise ValueError("Choose a locally installed model, not a cloud model.")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid local Ollama port.")
        if context != 4096:
            raise ValueError("This demo uses a fixed 4096-token context.")
        self.model, self.port, self.context = model, port, context

    def _request(self, path, payload=None, *, timeout=180):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            body = None if payload is None else json.dumps(payload).encode("utf-8")
            connection.request(
                "GET" if payload is None else "POST",
                path,
                body,
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw = response.read(262145)
            if response.status != 200 or len(raw) > 262144:
                raise OllamaError("Ollama returned an error or an oversized response.")
            result = json.loads(raw)
            if not isinstance(result, dict) or "error" in result:
                raise OllamaError("Ollama returned an invalid response.")
            return result
        except (OSError, http.client.HTTPException, ValueError) as error:
            raise OllamaError(
                "Cannot read the local Ollama response. Check the server, model and timeout."
            ) from error
        finally:
            connection.close()

    def identity(self):
        """Require an installed local model with tools; never pull as a side effect."""
        version = self._request("/api/version", timeout=10).get("version")
        tags = self._request("/api/tags", timeout=10)
        models = tags.get("models", [])
        if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
            raise OllamaError("Ollama returned an invalid installed-model list.")
        match = next((item for item in models if item.get("name") == self.model), None)
        if not match or not match.get("digest"):
            raise OllamaError("Model is not installed locally. Pull it explicitly with Ollama.")
        details = self._request("/api/show", {"model": self.model}, timeout=30)
        if details.get("remote_host") or details.get("remote_model"):
            raise OllamaError("Remote-backed models are not allowed by this local adapter.")
        if "tools" not in details.get("capabilities", []):
            raise OllamaError("The installed model does not advertise tool calling.")
        return {"model": self.model, "digest": match["digest"], "ollama_version": version}

    def unload(self):
        """Release this model after a demo; this request does not generate text."""
        self._request("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=20)

    def chat(self, messages, *, tools=None, schema=None, timeout=180):
        output_limit = 768 if schema else 512
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": "30s",
            "options": {
                "num_ctx": self.context,
                "num_predict": output_limit,
                "temperature": 0,
                "presence_penalty": 0,
                "seed": 42,
            },
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": item} for item in tools]
        if schema:
            payload["format"] = schema
        # Small prompts and a separate synthesis request keep this demo within laptop limits.
        if len(json.dumps(payload, ensure_ascii=False)) > 14000:
            raise OllamaError("Request exceeds the local prompt character budget.")
        response = self._request("/api/chat", payload, timeout=timeout)
        if response.get("done") is not True or response.get("done_reason") == "length":
            raise OllamaError("Model response was incomplete or reached its output limit.")
        token_count = response.get("prompt_eval_count", 0)
        if type(token_count) is not int or token_count < 0:
            raise OllamaError("Ollama returned invalid token usage.")
        if token_count >= self.context - output_limit:
            raise OllamaError("Prompt left insufficient context headroom.")
        message = response.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise OllamaError("Ollama did not return an assistant message.")
        if not isinstance(message.get("content", ""), str):
            raise OllamaError("Assistant content must be text.")
        return response
