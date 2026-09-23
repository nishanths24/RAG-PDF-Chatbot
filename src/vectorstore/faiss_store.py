"""Persistent FAISS vector store for the RAG PDF chatbot."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


DEFAULT_VECTORSTORE_DIR = Path("data/vectorstore")
DEFAULT_INDEX_FILENAME = "index.faiss"
DEFAULT_METADATA_FILENAME = "metadata.json"


class VectorStoreError(RuntimeError):
    """Base exception for vector-store errors."""


class VectorStoreDimensionError(VectorStoreError):
    """Raised when embedding dimensions do not match the FAISS index."""


class VectorStorePersistenceError(VectorStoreError):
    """Raised when persisted vector-store data is invalid."""


class FAISSVectorStore:
    """Persistent FAISS vector store using normalized inner-product search."""

    def __init__(
        self,
        dimension: int = 384,
        persist_directory: str | Path = DEFAULT_VECTORSTORE_DIR,
    ) -> None:
        """Initialize a FAISS vector store.

        Args:
            dimension: Embedding vector dimension.
            persist_directory: Directory used for persistent storage.
        """
        if dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero.")

        self.dimension = int(dimension)
        self.persist_directory = Path(persist_directory)

        self.index_path = self.persist_directory / DEFAULT_INDEX_FILENAME
        self.metadata_path = self.persist_directory / DEFAULT_METADATA_FILENAME

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._documents: list[Document] = []
        self._index = self._create_empty_index()

    def _create_empty_index(self) -> faiss.Index:
        """Create an empty FAISS inner-product index."""
        return faiss.IndexFlatIP(self.dimension)

    @staticmethod
    def _document_to_metadata(document: Document) -> dict:
        """Convert a LangChain Document into JSON-safe metadata."""
        return {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }

    @staticmethod
    def _metadata_to_document(data: dict) -> Document:
        """Convert persisted data back into a LangChain Document."""
        if "page_content" not in data:
            raise VectorStorePersistenceError(
                "Persisted document is missing page_content."
            )

        metadata = data.get("metadata", {})

        if not isinstance(metadata, dict):
            raise VectorStorePersistenceError(
                "Persisted document metadata must be a JSON object."
            )

        return Document(
            page_content=str(data["page_content"]),
            metadata=metadata,
        )

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Add documents and their embeddings to the FAISS index.

        Args:
            documents: LangChain Document objects.
            embeddings: Corresponding embedding vectors.

        Raises:
            ValueError: For empty input or mismatched document/vector counts.
            VectorStoreDimensionError: For incorrect embedding dimensions.
        """
        if not documents:
            raise ValueError("At least one document is required.")

        if not embeddings:
            raise ValueError("At least one embedding is required.")

        if len(documents) != len(embeddings):
            raise ValueError(
                "The number of documents must match the number of embeddings."
            )

        for document in documents:
            if not isinstance(document, Document):
                raise TypeError(
                    "All documents must be LangChain Document objects."
                )

            if not document.page_content.strip():
                raise ValueError(
                    "Documents must not contain empty page content."
                )

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must form a 2-dimensional array."
            )

        if vectors.shape[1] != self.dimension:
            raise VectorStoreDimensionError(
                f"Expected embedding dimension {self.dimension}, "
                f"received {vectors.shape[1]}."
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                "Embeddings must contain only finite values."
            )

        self._index.add(vectors)
        self._documents.extend(documents)

        logger.info(
            "Added %d documents to FAISS vector store. Total: %d",
            len(documents),
            self.count(),
        )

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        """Return documents ranked by inner-product similarity.

        Args:
            query_embedding: Query embedding vector.
            k: Maximum number of results.

        Returns:
            List of (Document, similarity_score) tuples.
        """
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        if self.is_empty():
            return []

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim != 1:
            raise ValueError(
                "Query embedding must be a one-dimensional vector."
            )

        if query.shape[0] != self.dimension:
            raise VectorStoreDimensionError(
                f"Expected query dimension {self.dimension}, "
                f"received {query.shape[0]}."
            )

        if not np.isfinite(query).all():
            raise ValueError(
                "Query embedding must contain only finite values."
            )

        query = query.reshape(1, -1)

        result_count = min(k, self.count())
        scores, indices = self._index.search(
            query,
            result_count,
        )

        results: list[tuple[Document, float]] = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            results.append(
                (
                    self._documents[int(index)],
                    float(score),
                )
            )

        return results

    def save(self) -> None:
        """Persist the FAISS index and document metadata to disk."""
        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            faiss.write_index(
                self._index,
                str(self.index_path),
            )

            metadata = [
                self._document_to_metadata(document)
                for document in self._documents
            ]

            with self.metadata_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    metadata,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            logger.info(
                "Saved FAISS vector store with %d vectors.",
                self.count(),
            )

        except (OSError, TypeError, ValueError) as exc:
            raise VectorStorePersistenceError(
                "Failed to save the FAISS vector store."
            ) from exc

    def load(self) -> None:
        """Load the FAISS index and metadata from disk.

        If no persisted files exist, the store remains empty.
        """
        index_exists = self.index_path.exists()
        metadata_exists = self.metadata_path.exists()

        if not index_exists and not metadata_exists:
            logger.info(
                "No persisted FAISS vector store found. "
                "Starting empty."
            )

            self._index = self._create_empty_index()
            self._documents = []

            return

        if index_exists != metadata_exists:
            raise VectorStorePersistenceError(
                "FAISS index and metadata files must both exist."
            )

        try:
            index = faiss.read_index(
                str(self.index_path)
            )

            if index.d != self.dimension:
                raise VectorStoreDimensionError(
                    f"Persisted index dimension is {index.d}; "
                    f"expected {self.dimension}."
                )

            with self.metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)

            if not isinstance(metadata, list):
                raise VectorStorePersistenceError(
                    "Persisted metadata must be a JSON list."
                )

            documents = [
                self._metadata_to_document(item)
                for item in metadata
            ]

            if index.ntotal != len(documents):
                raise VectorStorePersistenceError(
                    "FAISS index count does not match metadata count."
                )

            self._index = index
            self._documents = documents

            logger.info(
                "Loaded FAISS vector store with %d vectors.",
                self.count(),
            )

        except VectorStoreError:
            raise

        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            raise VectorStorePersistenceError(
                "Failed to load the FAISS vector store."
            ) from exc

    def clear(self) -> None:
        """Clear the vector store and remove persisted files."""
        self._index = self._create_empty_index()
        self._documents = []

        for path in (
            self.index_path,
            self.metadata_path,
        ):
            try:
                if path.exists():
                    path.unlink()

            except OSError as exc:
                raise VectorStorePersistenceError(
                    f"Failed to remove persisted file: {path}"
                ) from exc

        logger.info("FAISS vector store cleared.")

    def count(self) -> int:
        """Return the number of indexed vectors."""
        return int(self._index.ntotal)

    def contains_document_id(self, document_id: str) -> bool:
        """Return True when a document ID is already indexed."""
        cleaned_document_id = str(document_id).strip()

        if not cleaned_document_id:
            raise ValueError(
                "document_id must not be empty."
            )

        return any(
            str(
                document.metadata.get(
                    "document_id",
                    "",
                )
            ).strip()
            == cleaned_document_id
            for document in self._documents
        )

    def is_empty(self) -> bool:
        """Return True when the vector store contains no vectors."""
        return self.count() == 0