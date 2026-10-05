"""Groq LLM service."""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_GROQ_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.1-8b-instant"


class GroqServiceError(RuntimeError):
    """Raised when communication with Groq fails."""


class GroqService:
    """Generate text using the Groq API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_GROQ_URL,
        timeout: int = 120,
        temperature: float = 0.0,
    ) -> None:
        load_dotenv()

        if not model.strip():
            raise ValueError("Groq model must not be empty.")

        if not base_url.strip():
            raise ValueError("Groq base URL must not be empty.")

        if timeout <= 0:
            raise ValueError("Groq timeout must be greater than zero.")

        if temperature < 0:
            raise ValueError("Temperature must not be negative.")

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise GroqServiceError(
                "GROQ_API_KEY is not configured. "
                "Add it to the .env file."
            )

        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        """Generate a response from the Groq model."""

        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            raise ValueError("Prompt must not be empty.")

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": cleaned_prompt,
                        }
                    ],
                    "temperature": self.temperature,
                    "stream": False,
                },
                timeout=self.timeout,
            )

        except requests.RequestException as exc:
            raise GroqServiceError(
                "Unable to connect to Groq. "
                "Check your internet connection and Groq API availability."
            ) from exc

        if response.status_code != 200:
            raise GroqServiceError(
                f"Groq returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise GroqServiceError(
                "Groq returned an invalid JSON response."
            ) from exc

        try:
            generated_text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GroqServiceError(
                "Groq response did not contain generated text."
            ) from exc

        if not isinstance(generated_text, str):
            raise GroqServiceError(
                "Groq response content was not text."
            )

        return generated_text.strip()

    def is_available(self) -> bool:
        """Return True when the Groq API is reachable."""

        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=10,
            )

            return response.status_code == 200

        except requests.RequestException:
            return False