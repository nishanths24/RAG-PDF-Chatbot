"""Tests for the RAG retrieval layer."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from langchain_core.documents import Document

from src.retrieval.retriever import RetrievalResult, Retriever


class RetrieverTests(unittest.TestCase):
    """Test query embedding and FAISS retrieval integration."""

    def setUp(self) -> None:
        self.embedding_service = Mock()

        self.vector_store = Mock()

        self.documents = [
            Document(
                page_content="Machine learning learns patterns from data.",
                metadata={
                    "source": "ml.pdf",
                    "page": 1,
                    "document_id": "doc-ml",
                },
            ),
            Document(
                page_content="Deep learning uses multi-layer neural networks.",
                metadata={
                    "source": "dl.pdf",
                    "page": 2,
                    "document_id": "doc-dl",
                },
            ),
        ]

        self.embedding_service.embed_query.return_value = [
            1.0,
            0.0,
            0.0,
        ]

        self.vector_store.similarity_search.return_value = [
            (self.documents[0], 0.95),
            (self.documents[1], 0.72),
        ]

        self.vector_store.count.return_value = 2
        self.vector_store.is_empty.return_value = False

        self.retriever = Retriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            top_k=2,
        )

    def test_retrieve_embeds_query(self) -> None:
        results = self.retriever.retrieve(
            "What is machine learning?"
        )

        self.embedding_service.embed_query.assert_called_once_with(
            "What is machine learning?"
        )

        self.assertEqual(len(results), 2)

    def test_retrieve_calls_vector_store_with_top_k(self) -> None:
        self.retriever.retrieve("What is machine learning?")

        self.vector_store.similarity_search.assert_called_once_with(
            [1.0, 0.0, 0.0],
            k=2,
        )

    def test_retrieve_returns_retrieval_results(self) -> None:
        results = self.retriever.retrieve(
            "What is machine learning?"
        )

        self.assertIsInstance(results[0], RetrievalResult)
        self.assertEqual(
            results[0].document,
            self.documents[0],
        )
        self.assertAlmostEqual(results[0].score, 0.95)

    def test_retrieve_documents_returns_documents_only(self) -> None:
        results = self.retriever.retrieve_documents(
            "What is machine learning?"
        )

        self.assertEqual(results, self.documents)

    def test_retrieve_with_scores_returns_tuples(self) -> None:
        results = self.retriever.retrieve_with_scores(
            "What is machine learning?"
        )

        self.assertEqual(
            results,
            [
                (self.documents[0], 0.95),
                (self.documents[1], 0.72),
            ],
        )

    def test_score_threshold_filters_results(self) -> None:
        retriever = Retriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            top_k=2,
            score_threshold=0.80,
        )

        results = retriever.retrieve(
            "What is machine learning?"
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].document,
            self.documents[0],
        )

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.retrieve("   ")

        self.embedding_service.embed_query.assert_not_called()

    def test_top_k_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            Retriever(
                embedding_service=self.embedding_service,
                vector_store=self.vector_store,
                top_k=0,
            )

    def test_invalid_score_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Retriever(
                embedding_service=self.embedding_service,
                vector_store=self.vector_store,
                score_threshold=1.5,
            )

    def test_count_delegates_to_vector_store(self) -> None:
        self.assertEqual(self.retriever.count(), 2)
        self.vector_store.count.assert_called_once()

    def test_is_empty_delegates_to_vector_store(self) -> None:
        self.assertFalse(self.retriever.is_empty())
        self.vector_store.is_empty.assert_called_once()

    def test_empty_vector_store_returns_no_results(self) -> None:
        self.vector_store.similarity_search.return_value = []

        results = self.retriever.retrieve(
            "What is machine learning?"
        )

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()