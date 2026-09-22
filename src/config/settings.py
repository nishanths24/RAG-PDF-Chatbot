"""Typed application settings loaded from YAML and environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
YAML_CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"


class ApplicationSettings(BaseModel):
    """High-level application metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str
    environment: str

    @field_validator("name", "environment")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class LLMSettings(BaseModel):
    """Large language model configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    temperature: float = Field(ge=0.0, le=2.0)

    @field_validator("provider", "model")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class EmbeddingSettings(BaseModel):
    """Embedding model configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str

    @field_validator("provider", "model")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class RetrievalSettings(BaseModel):
    """Retriever configuration."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(ge=1, le=50)


class ChunkingSettings(BaseModel):
    """Document chunking configuration."""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_within_chunk_size(self) -> ChunkingSettings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class PathSettings(BaseModel):
    """Project-relative data paths."""

    model_config = ConfigDict(extra="forbid")

    uploads: str
    vectorstore: str

    @field_validator("uploads", "vectorstore")
    @classmethod
    def _relative_posix_path(cls, value: str) -> str:
        cleaned = value.strip().replace("\\", "/")
        if not cleaned:
            raise ValueError("path must not be blank")
        path = Path(cleaned)
        if path.is_absolute() or (len(cleaned) >= 2 and cleaned[1] == ":"):
            raise ValueError("paths must be relative to the project root")
        if ".." in path.parts:
            raise ValueError("paths must not contain parent-directory segments")
        return cleaned


class Settings(BaseModel):
    """Complete application settings for RAG PDF Chatbot."""

    model_config = ConfigDict(extra="forbid")

    application: ApplicationSettings
    llm: LLMSettings
    embeddings: EmbeddingSettings
    retrieval: RetrievalSettings
    chunking: ChunkingSettings
    paths: PathSettings
    openai_api_key: str = ""

    @property
    def project_root(self) -> Path:
        """Return the repository root directory."""
        return PROJECT_ROOT

    @property
    def uploads_dir(self) -> Path:
        """Absolute path to the PDF upload directory."""
        return (PROJECT_ROOT / self.paths.uploads).resolve()

    @property
    def vectorstore_dir(self) -> Path:
        """Absolute path to the FAISS vectorstore directory."""
        return (PROJECT_ROOT / self.paths.vectorstore).resolve()

    def ensure_data_directories(self) -> None:
        """Create configured data directories if they do not exist."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


def _load_env() -> None:
    """Load variables from the project `.env` file if it exists."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load and validate the YAML configuration mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping")

    required_sections = (
        "application",
        "llm",
        "embeddings",
        "retrieval",
        "chunking",
        "paths",
    )
    missing = [section for section in required_sections if section not in data]
    if missing:
        raise ValueError(f"Missing required configuration sections: {', '.join(missing)}")
    return data


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Overlay environment-specific model names onto YAML defaults."""
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "").strip()
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip()

    if chat_model:
        raw.setdefault("llm", {})
        raw["llm"]["model"] = chat_model
    if embedding_model:
        raw.setdefault("embeddings", {})
        raw["embeddings"]["model"] = embedding_model
    return raw


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load, validate, and cache application settings.

    Environment variables overlay YAML defaults for model names.
    An OpenAI API key is exposed when present but is not required to boot
    Phase 1 of the application.
    """
    _load_env()
    raw = _apply_env_overrides(_load_yaml_config(YAML_CONFIG_FILE))
    settings = Settings(
        **raw,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
    )
    settings.ensure_data_directories()
    return settings
