"""Local embedding service for document and query vectors."""

from __future__ import annotations

from typing import Sequence

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingConfigurationError(RuntimeError):
    """Raised when local embedding configuration is invalid."""


class EmbeddingService:
    """Provide local Sentence Transformer embeddings."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the local embedding model.

        Args:
            model: Optional Sentence Transformer model name.

        Raises:
            EmbeddingConfigurationError: If the model cannot be loaded.
        """
        self.model_name = model or DEFAULT_EMBEDDING_MODEL

        try:
            self._embeddings = SentenceTransformer(
                self.model_name,
                device="cpu",
            )
        except Exception as exc:
            raise EmbeddingConfigurationError(
                f"Unable to load embedding model '{self.model_name}'."
            ) from exc

    @property
    def embeddings(self) -> SentenceTransformer:
        """Return the underlying Sentence Transformer model."""
        return self._embeddings

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple document chunks.

        Args:
            texts: Text chunks to embed.

        Returns:
            One normalized embedding vector per input text.

        Raises:
            ValueError: If the input is empty or contains empty strings.
        """
        if not texts:
            raise ValueError("At least one document text is required.")

        cleaned_texts = [str(text).strip() for text in texts]

        if any(not text for text in cleaned_texts):
            raise ValueError("Document texts must not contain empty strings.")

        vectors = self._embeddings.encode(
            cleaned_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a user query.

        Args:
            text: User query.

        Returns:
            Normalized query embedding vector.

        Raises:
            ValueError: If the query is empty.
        """
        cleaned_text = str(text).strip()

        if not cleaned_text:
            raise ValueError("Query text must not be empty.")

        vector = self._embeddings.encode(
            cleaned_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vector.tolist()

    def get_embedding_dimension(self) -> int:
        """Return the local embedding vector dimension."""
        return int(self._embeddings.get_sentence_embedding_dimension())