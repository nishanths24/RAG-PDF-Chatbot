"""Query retrieval layer for the RAG PDF chatbot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document

from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.faiss_store import FAISSVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document chunk and its similarity score."""

    document: Document
    score: float


class Retriever:
    """Retrieve relevant document chunks using local embeddings and FAISS."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: FAISSVectorStore | None = None,
        top_k: int = 4,
        score_threshold: float | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            embedding_service: Local embedding service.
            vector_store: Persistent FAISS vector store.
            top_k: Maximum number of chunks to retrieve.
            score_threshold: Optional minimum similarity score.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if score_threshold is not None and not (
            -1.0 <= score_threshold <= 1.0
        ):
            raise ValueError(
                "score_threshold must be between -1.0 and 1.0."
            )

        self.embedding_service = (
            embedding_service or EmbeddingService()
        )

        self.vector_store = vector_store or FAISSVectorStore(
            dimension=self.embedding_service.get_embedding_dimension()
        )

        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Retrieve the most relevant document chunks for a query.

        Args:
            query: User's natural-language question.

        Returns:
            Retrieved document chunks ordered by similarity.

        Raises:
            ValueError: If the query is empty.
        """
        cleaned_query = str(query).strip()

        if not cleaned_query:
            raise ValueError("Query must not be empty.")

        query_embedding = self.embedding_service.embed_query(
            cleaned_query
        )

        results = self.vector_store.similarity_search(
            query_embedding,
            k=self.top_k,
        )

        retrieved = [
            RetrievalResult(
                document=document,
                score=score,
            )
            for document, score in results
        ]

        if self.score_threshold is not None:
            retrieved = [
                result
                for result in retrieved
                if result.score >= self.score_threshold
            ]

        return retrieved

    def retrieve_documents(self, query: str) -> list[Document]:
        """Retrieve only the matching LangChain Documents."""
        return [
            result.document
            for result in self.retrieve(query)
        ]

    def retrieve_with_scores(
        self,
        query: str,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents together with their similarity scores."""
        return [
            (result.document, result.score)
            for result in self.retrieve(query)
        ]

    def count(self) -> int:
        """Return the number of vectors currently available."""
        return self.vector_store.count()

    def is_empty(self) -> bool:
        """Return whether the underlying vector store is empty."""
        return self.vector_store.is_empty()