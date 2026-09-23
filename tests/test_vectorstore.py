"""Tests for the persistent FAISS vector store."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from src.vectorstore.faiss_store import (
    FAISSVectorStore,
    VectorStoreDimensionError,
    VectorStorePersistenceError,
)


EMBEDDING_DIMENSION = 384


def make_vector(index: int) -> list[float]:
    """Create a deterministic 384-dimensional test vector."""
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[index] = 1.0
    return vector


class FAISSVectorStoreTests(unittest.TestCase):
    """Test the persistent FAISS vector store."""

    def setUp(self) -> None:
        """Create an isolated temporary vector-store directory."""
        self.temp_directory = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_directory.name)

        self.store = FAISSVectorStore(
            dimension=EMBEDDING_DIMENSION,
            persist_directory=self.store_path,
        )

        self.documents = [
            Document(
                page_content="Machine learning is a branch of artificial intelligence.",
                metadata={
                    "source": "machine_learning.pdf",
                    "page": 1,
                    "document_id": "doc-001",
                },
            ),
            Document(
                page_content="Deep learning uses neural networks with multiple layers.",
                metadata={
                    "source": "deep_learning.pdf",
                    "page": 2,
                    "document_id": "doc-002",
                },
            ),
        ]

        self.embeddings = [
            make_vector(0),
            make_vector(1),
        ]

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        self.temp_directory.cleanup()

    def test_initial_store_is_empty(self) -> None:
        self.assertTrue(self.store.is_empty())
        self.assertEqual(self.store.count(), 0)

    def test_add_documents(self) -> None:
        self.store.add_documents(
            self.documents,
            self.embeddings,
        )

        self.assertEqual(self.store.count(), 2)
        self.assertFalse(self.store.is_empty())

    def test_similarity_search_returns_best_match(self) -> None:
        self.store.add_documents(
            self.documents,
            self.embeddings,
        )

        results = self.store.similarity_search(
            make_vector(0),
            k=1,
        )

        self.assertEqual(len(results), 1)

        document, score = results[0]

        self.assertEqual(
            document.page_content,
            self.documents[0].page_content,
        )
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_similarity_search_does_not_exceed_available_documents(self) -> None:
        self.store.add_documents(
            self.documents,
            self.embeddings,
        )

        results = self.store.similarity_search(
            make_vector(0),
            k=10,
        )

        self.assertEqual(len(results), 2)

    def test_metadata_is_preserved(self) -> None:
        self.store.add_documents(
            self.documents,
            self.embeddings,
        )

        results = self.store.similarity_search(
            make_vector(1),
            k=1,
        )

        document, _ = results[0]

        self.assertEqual(
            document.metadata["source"],
            "deep_learning.pdf",
        )
        self.assertEqual(
            document.metadata["page"],
            2,
        )
        self.assertEqual(
            document.metadata["document_id"],
            "doc-002",
        )

    def test_save_and_load_persistence(self) -> None:
        self.store.add_documents(
            self.documents,
            self.embeddings,
        )

        self.store.save()

        self.assertTrue(
            (self.store_path / "index.faiss").exists()
        )
        self.assertTrue(
            (self.store_path / "metadata.json").exists()
        )

        loaded_store = FAISSVectorStore(
            dimension=EMBEDDING_DIMENSION,
            persist_directory=self.store_path,
        )

        loaded_store.load()

        self.assertEqual(loaded_store.count(), 2)

        results = loaded_store.similarity_search(
            make_vector(0),
            k=1,
        )

        document, score = results[0]

        self.assertEqual(
            document.page_content,
            self.documents[0].page_content,
        )
        self.assertEqual(
            document.metadata["document_id"],
            "doc-001",
        )
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_load_missing_store_starts_empty(self) -> None:
        loaded_store = FAISSVectorStore(
            dimension=EMBEDDING_DIMENSION,
            persist_directory=self.store_path,
        )

        loaded_store.load()

        self.assertTrue(loaded_store.is_empty())
        self.assertEqual(loaded_store.count(), 0)

    def test_clear_removes_vectors_and_persistence(self) -> None:
        self.store.add_documents(
            self.documents,
            self.embeddings,
        )
        self.store.save()

        self.store.clear()

        self.assertTrue(self.store.is_empty())
        self.assertEqual(self.store.count(), 0)

        self.assertFalse(
            (self.store_path / "index.faiss").exists()
        )
        self.assertFalse(
            (self.store_path / "metadata.json").exists()
        )

    def test_document_embedding_count_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add_documents(
                self.documents,
                [self.embeddings[0]],
            )

    def test_embedding_dimension_mismatch(self) -> None:
        invalid_embedding = [0.0] * 128

        with self.assertRaises(VectorStoreDimensionError):
            self.store.add_documents(
                [self.documents[0]],
                [invalid_embedding],
            )

    def test_query_dimension_mismatch(self) -> None:
        invalid_query = [0.0] * 128

        self.store.add_documents(
            self.documents,
            self.embeddings,
        )

        with self.assertRaises(VectorStoreDimensionError):
            self.store.similarity_search(
                invalid_query,
                k=1,
            )

    def test_empty_document_is_rejected(self) -> None:
        empty_document = Document(
            page_content="",
            metadata={},
        )

        with self.assertRaises(ValueError):
            self.store.add_documents(
                [empty_document],
                [make_vector(0)],
            )

    def test_invalid_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.similarity_search(
                make_vector(0),
                k=0,
            )

    def test_similarity_search_on_empty_store(self) -> None:
        results = self.store.similarity_search(
            make_vector(0),
            k=4,
        )

        self.assertEqual(results, [])

    def test_partial_persistence_files_raise_error(self) -> None:
        self.store.add_documents(
            self.documents,
            self.embeddings,
        )
        self.store.save()

        (self.store_path / "metadata.json").unlink()

        loaded_store = FAISSVectorStore(
            dimension=EMBEDDING_DIMENSION,
            persist_directory=self.store_path,
        )

        with self.assertRaises(VectorStorePersistenceError):
            loaded_store.load()


if __name__ == "__main__":
    unittest.main()