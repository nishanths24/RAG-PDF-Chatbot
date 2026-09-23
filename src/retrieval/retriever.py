"""Document retrieval for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.faiss_store import FAISSVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document and its similarity score."""

    document: Document
    score: float


class Retriever:
    """Retrieve relevant document chunks from the vector store."""

    BROAD_QUERY_PHRASES = (
        "summarize",
        "summary",
        "summarise",
        "overview",
        "what is this document",
        "what is the document about",
        "what is this pdf",
        "what is the pdf about",
        "main topics",
        "key topics",
        "main points",
        "key points",
        "give me an overview",
        "describe this document",
    )

    BROAD_QUERY_TEMPLATES = (
        "What is the main topic and purpose of this document?",
        "What are the main topics, sections, and concepts discussed in this document?",
        "What are the key objectives, methods, strategies, or features discussed in this document?",
        "What are the important findings, results, conclusions, or takeaways in this document?",
        "What are the most important ideas a reader should know from this document?",
    )

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        *,
        top_k: int = 4,
        score_threshold: float | None = None,
    ) -> None:
        """Initialize the retriever."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if score_threshold is not None and not -1.0 <= score_threshold <= 1.0:
            raise ValueError(
                "score_threshold must be between -1.0 and 1.0."
            )

        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def _is_broad_question(self, query: str) -> bool:
        """Return whether the query asks for document-level information."""

        normalized = " ".join(query.lower().split())

        return any(
            phrase in normalized
            for phrase in self.BROAD_QUERY_PHRASES
        )

    def _apply_score_threshold(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Apply the optional similarity threshold."""

        if self.score_threshold is None:
            return results

        return [
            result
            for result in results
            if result.score >= self.score_threshold
        ]

    def _retrieve_standard(
        self,
        query: str,
    ) -> list[RetrievalResult]:
        """Perform normal single-query semantic retrieval."""

        query_embedding = self.embedding_service.embed_query(query)

        matches = self.vector_store.similarity_search(
            query_embedding,
            k=self.top_k,
        )

        results = [
            RetrievalResult(
                document=document,
                score=float(score),
            )
            for document, score in matches
        ]

        return self._apply_score_threshold(results)

    def _retrieve_broad(
        self,
        query: str,
    ) -> list[RetrievalResult]:
        """Retrieve diverse context for broad document-level questions."""

        retrieval_queries = [
            query,
            *self.BROAD_QUERY_TEMPLATES,
        ]

        candidate_k = max(2, self.top_k)

        candidates: dict[str, RetrievalResult] = {}

        for retrieval_query in retrieval_queries:
            query_embedding = self.embedding_service.embed_query(
                retrieval_query
            )

            matches = self.vector_store.similarity_search(
                query_embedding,
                k=candidate_k,
            )

            for document, score in matches:
                metadata = document.metadata or {}

                chunk_id = str(
                    metadata.get(
                        "chunk_id",
                        (
                            f"{metadata.get('file_name', '')}"
                            f"-{metadata.get('page', '')}"
                            f"-{document.page_content[:80]}"
                        ),
                    )
                )

                result = RetrievalResult(
                    document=document,
                    score=float(score),
                )

                existing = candidates.get(chunk_id)

                if existing is None or result.score > existing.score:
                    candidates[chunk_id] = result

        ranked_candidates = sorted(
            candidates.values(),
            key=lambda result: result.score,
            reverse=True,
        )

        return self._apply_score_threshold(
            ranked_candidates[: self.top_k]
        )

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a user query."""

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("Query must not be empty.")

        if self.vector_store.is_empty():
            return []

        if self._is_broad_question(cleaned_query):
            return self._retrieve_broad(cleaned_query)

        return self._retrieve_standard(cleaned_query)

    def retrieve_documents(
        self,
        query: str,
    ) -> list[Document]:
        """Retrieve documents without similarity scores."""

        return [
            result.document
            for result in self.retrieve(query)
        ]

    def retrieve_with_scores(
        self,
        query: str,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with similarity scores."""

        return [
            (result.document, result.score)
            for result in self.retrieve(query)
        ]

    def count(self) -> int:
        """Return the number of indexed chunks."""

        return self.vector_store.count()

    def is_empty(self) -> bool:
        """Return whether the vector store is empty."""

        return self.vector_store.is_empty()