"""Tests for the document indexing service."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from src.services.document_service import (
    DocumentService,
    IndexingResult,
)


class DocumentServiceTests(unittest.TestCase):
    """Test document indexing orchestration."""

    def setUp(self) -> None:
        self.embedding_service = Mock()
        self.vector_store = Mock()

        self.service = DocumentService(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
        )

    @patch("src.services.document_service.split_documents")
    @patch("src.services.document_service.process_pdf")
    def test_index_pdf_runs_complete_pipeline(
        self,
        mock_process_pdf: Mock,
        mock_split_documents: Mock,
    ) -> None:
        page_documents = [
            Document(
                page_content="Page one content.",
                metadata={
                    "file_name": "sample.pdf",
                    "page": 1,
                    "document_id": "abc123",
                },
            ),
            Document(
                page_content="Page two content.",
                metadata={
                    "file_name": "sample.pdf",
                    "page": 2,
                    "document_id": "abc123",
                },
            ),
        ]

        chunks = [
            Document(
                page_content="Page one content.",
                metadata={
                    "file_name": "sample.pdf",
                    "page": 1,
                    "document_id": "abc123",
                    "chunk_id": "abc123-p1-c0",
                },
            ),
            Document(
                page_content="Page two content.",
                metadata={
                    "file_name": "sample.pdf",
                    "page": 2,
                    "document_id": "abc123",
                    "chunk_id": "abc123-p2-c0",
                },
            ),
        ]

        embeddings = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

        mock_process_pdf.return_value = page_documents
        mock_split_documents.return_value = chunks
        self.embedding_service.embed_documents.return_value = embeddings

        result = self.service.index_pdf("sample.pdf")

        self.assertIsInstance(result, IndexingResult)
        self.assertEqual(result.file_name, "sample.pdf")
        self.assertEqual(result.document_id, "abc123")
        self.assertEqual(result.pages, 2)
        self.assertEqual(result.chunks, 2)
        self.assertEqual(result.vectors, 2)

        mock_process_pdf.assert_called_once_with("sample.pdf")
        mock_split_documents.assert_called_once_with(
            page_documents,
            chunk_size=1000,
            chunk_overlap=150,
        )
        self.embedding_service.embed_documents.assert_called_once_with(
            [
                "Page one content.",
                "Page two content.",
            ]
        )
        self.vector_store.add_documents.assert_called_once_with(
            chunks,
            embeddings,
        )
        self.vector_store.save.assert_called_once()

    @patch("src.services.document_service.split_documents")
    @patch("src.services.document_service.process_pdf")
    def test_index_pdf_rejects_empty_pages(
        self,
        mock_process_pdf: Mock,
        mock_split_documents: Mock,
    ) -> None:
        mock_process_pdf.return_value = []

        with self.assertRaises(ValueError):
            self.service.index_pdf("empty.pdf")

        mock_split_documents.assert_not_called()
        self.embedding_service.embed_documents.assert_not_called()
        self.vector_store.add_documents.assert_not_called()

    @patch("src.services.document_service.split_documents")
    @patch("src.services.document_service.process_pdf")
    def test_index_pdf_rejects_empty_chunks(
        self,
        mock_process_pdf: Mock,
        mock_split_documents: Mock,
    ) -> None:
        mock_process_pdf.return_value = [
            Document(
                page_content="Some content.",
                metadata={
                    "file_name": "sample.pdf",
                    "page": 1,
                    "document_id": "abc123",
                },
            )
        ]

        mock_split_documents.return_value = []

        with self.assertRaises(ValueError):
            self.service.index_pdf("sample.pdf")

        self.embedding_service.embed_documents.assert_not_called()
        self.vector_store.add_documents.assert_not_called()

    def test_load_existing_index_delegates_to_vector_store(self) -> None:
        self.service.load_existing_index()

        self.vector_store.load.assert_called_once()

    def test_clear_index_delegates_to_vector_store(self) -> None:
        self.service.clear_index()

        self.vector_store.clear.assert_called_once()

    def test_get_indexed_document_count_delegates(self) -> None:
        self.vector_store.count.return_value = 25

        result = self.service.get_indexed_document_count()

        self.assertEqual(result, 25)
        self.vector_store.count.assert_called_once()

    def test_is_index_empty_delegates(self) -> None:
        self.vector_store.is_empty.return_value = True

        result = self.service.is_index_empty()

        self.assertTrue(result)
        self.vector_store.is_empty.assert_called_once()


if __name__ == "__main__":
    unittest.main()