"""Application configuration management."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YAML_CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class ApplicationSettings:
    """General application settings."""

    name: str
    environment: str


@dataclass(frozen=True)
class LLMSettings:
    """Local LLM settings."""

    provider: str
    model: str
    temperature: float


@dataclass(frozen=True)
class EmbeddingSettings:
    """Embedding configuration."""

    provider: str
    model: str


@dataclass(frozen=True)
class RetrievalSettings:
    """Retrieval configuration."""

    top_k: int


@dataclass(frozen=True)
class ChunkingSettings:
    """Document chunking configuration."""

    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class PathSettings:
    """Application path configuration."""

    uploads: str
    vectorstore: str


@dataclass(frozen=True)
class Settings:
    """Complete application configuration."""

    application: ApplicationSettings
    llm: LLMSettings
    embeddings: EmbeddingSettings
    retrieval: RetrievalSettings
    chunking: ChunkingSettings
    paths: PathSettings

    @property
    def uploads_dir(self) -> Path:
        """Return the absolute PDF upload directory."""

        return PROJECT_ROOT / self.paths.uploads

    @property
    def vectorstore_dir(self) -> Path:
        """Return the absolute FAISS vector-store directory."""

        return PROJECT_ROOT / self.paths.vectorstore

    def ensure_data_directories(self) -> None:
        """Create required application directories."""

        self.uploads_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vectorstore_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


def _load_env() -> None:
    """Load environment variables from .env if present."""

    env_file = PROJECT_ROOT / ".env"

    if env_file.exists():
        load_dotenv(env_file)


def _load_yaml_config(
    config_path: Path,
) -> dict:
    """Load YAML configuration."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Configuration file must contain a YAML mapping."
        )

    return data


def _validate_settings(
    settings: Settings,
) -> None:
    """Validate application configuration."""

    if not settings.application.name.strip():
        raise ValueError(
            "Application name must not be empty."
        )

    if not settings.llm.provider.strip():
        raise ValueError(
            "LLM provider must not be empty."
        )

    if not settings.llm.model.strip():
        raise ValueError(
            "LLM model must not be empty."
        )

    if settings.llm.temperature < 0:
        raise ValueError(
            "LLM temperature must be >= 0."
        )

    if not settings.embeddings.provider.strip():
        raise ValueError(
            "Embedding provider must not be empty."
        )

    if not settings.embeddings.model.strip():
        raise ValueError(
            "Embedding model must not be empty."
        )

    if settings.retrieval.top_k <= 0:
        raise ValueError(
            "retrieval.top_k must be greater than zero."
        )

    if settings.chunking.chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if settings.chunking.chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap must be >= 0."
        )

    if (
        settings.chunking.chunk_overlap
        >= settings.chunking.chunk_size
    ):
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load, validate, and cache application settings."""

    _load_env()

    raw = _load_yaml_config(
        YAML_CONFIG_FILE
    )

    settings = Settings(
        application=ApplicationSettings(
            **raw["application"]
        ),
        llm=LLMSettings(
            **raw["llm"]
        ),
        embeddings=EmbeddingSettings(
            **raw["embeddings"]
        ),
        retrieval=RetrievalSettings(
            **raw["retrieval"]
        ),
        chunking=ChunkingSettings(
            **raw["chunking"]
        ),
        paths=PathSettings(
            **raw["paths"]
        ),
    )

    _validate_settings(settings)

    settings.ensure_data_directories()

    return settings