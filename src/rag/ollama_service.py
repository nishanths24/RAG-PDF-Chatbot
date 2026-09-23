"""Local Ollama LLM service."""

from __future__ import annotations

from typing import Any

import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"


class OllamaServiceError(RuntimeError):
    """Raised when communication with Ollama fails."""


class OllamaService:
    """Generate text using a locally running Ollama model."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = 120,
    ) -> None:
        if not model.strip():
            raise ValueError("Ollama model must not be empty.")

        if not base_url.strip():
            raise ValueError("Ollama base URL must not be empty.")

        if timeout <= 0:
            raise ValueError("Ollama timeout must be greater than zero.")

        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """Generate a response from the local Ollama model."""
        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            raise ValueError("Prompt must not be empty.")

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": cleaned_prompt,
                    "stream": False,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise OllamaServiceError(
                "Unable to connect to Ollama. "
                "Make sure Ollama is running."
            ) from exc

        if response.status_code != 200:
            raise OllamaServiceError(
                f"Ollama returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise OllamaServiceError(
                "Ollama returned an invalid JSON response."
            ) from exc

        generated_text = payload.get("response")

        if not isinstance(generated_text, str):
            raise OllamaServiceError(
                "Ollama response did not contain generated text."
            )

        return generated_text.strip()

    def is_available(self) -> bool:
        """Return True when the Ollama server is reachable."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False