"""Tests for the PDF document processor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from src.ingestion.document_processor import process_pdf


class DocumentProcessorTests(unittest.TestCase):
    """Validate PDF orchestration and document ID generation."""

    @patch("src.ingestion.document_processor.load_pdf")
    def test_process_pdf_generates_content_hash_and_passes_it_to_loader(
        self,
        mock_load_pdf: Mock,
    ) -> None:
        """Verify that the processor generates a SHA-256 document ID."""

        expected_documents = [
            Document(
                page_content="Test page",
                metadata={
                    "source": "sample.pdf",
                    "file_name": "sample.pdf",
                    "page": 1,
                    "document_id": "expected-hash",
                },
            )
        ]

        mock_load_pdf.return_value = expected_documents

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            pdf_path.write_bytes(b"PDF test content")

            documents = process_pdf(pdf_path)

        self.assertEqual(documents, expected_documents)

        mock_load_pdf.assert_called_once()

        called_path = mock_load_pdf.call_args.args[0]
        called_document_id = mock_load_pdf.call_args.kwargs["document_id"]

        self.assertEqual(
            Path(called_path),
            pdf_path,
        )

        self.assertEqual(
            len(called_document_id),
            64,
        )

        self.assertTrue(
            all(
                character in "0123456789abcdef"
                for character in called_document_id
            )
        )

    @patch("src.ingestion.document_processor.load_pdf")
    def test_process_pdf_can_resolve_upload_filename(
        self,
        mock_load_pdf: Mock,
    ) -> None:
        """Verify that a filename can be resolved from uploads_dir."""

        mock_load_pdf.return_value = [
            Document(
                page_content="Uploaded PDF",
                metadata={
                    "source": "uploaded.pdf",
                    "file_name": "uploaded.pdf",
                    "page": 1,
                    "document_id": "hash",
                },
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            pdf_path = upload_dir / "uploaded.pdf"
            pdf_path.write_bytes(b"uploaded PDF content")

            with patch(
                "src.ingestion.document_processor.get_settings"
            ) as mock_settings:
                settings = Mock()
                settings.uploads_dir = upload_dir
                mock_settings.return_value = settings

                documents = process_pdf("uploaded.pdf")

        self.assertEqual(len(documents), 1)
        mock_load_pdf.assert_called_once()


if __name__ == "__main__":
    unittest.main()