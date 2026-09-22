"""Tests for page-level PDF ingestion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pypdf import PdfWriter

from src.ingestion.exceptions import (
    EmptyPdfError,
    InvalidPdfPathError,
    NoExtractableTextError,
    PdfNotFoundError,
)
from src.ingestion.pdf_loader import load_pdf


class PdfLoaderTests(unittest.TestCase):
    """Validate PDF loading and page-level metadata."""

    def create_pdf(
        self,
        directory: Path,
        name: str = "sample.pdf",
        pages: int = 2,
    ) -> Path:
        """Create a minimal PDF for testing."""
        pdf_path = directory / name
        writer = PdfWriter()

        for _ in range(pages):
            writer.add_blank_page(width=300, height=300)

        with pdf_path.open("wb") as file:
            writer.write(file)

        return pdf_path

    def test_missing_file_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "missing.pdf"

            with self.assertRaises(PdfNotFoundError):
                load_pdf(
                    pdf_path,
                    document_id="test-document-id",
                )

    def test_non_pdf_file_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sample.txt"
            file_path.write_text(
                "not a PDF",
                encoding="utf-8",
            )

            with self.assertRaises(InvalidPdfPathError):
                load_pdf(
                    file_path,
                    document_id="test-document-id",
                )

    def test_empty_pdf_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "empty.pdf"
            pdf_path.touch()

            with self.assertRaises(EmptyPdfError):
                load_pdf(
                    pdf_path,
                    document_id="test-document-id",
                )

    def test_pdf_without_extractable_text_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = self.create_pdf(Path(temp_dir))

            with self.assertRaises(NoExtractableTextError):
                load_pdf(
                    pdf_path,
                    document_id="test-document-id",
                )

    @patch("src.ingestion.pdf_loader.PdfReader")
    def test_text_pages_create_documents_with_metadata(
        self,
        mock_pdf_reader: Mock,
    ) -> None:
        """Verify extracted pages become Documents with stable metadata."""

        mock_page_1 = Mock()
        mock_page_1.extract_text.return_value = "  First page text.  "

        mock_page_2 = Mock()
        mock_page_2.extract_text.return_value = "Second page text."

        mock_reader = Mock()
        mock_reader.pages = [
            mock_page_1,
            mock_page_2,
        ]
        mock_reader.is_encrypted = False

        mock_pdf_reader.return_value = mock_reader

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"

            # The loader only needs the file to exist for this mocked test.
            pdf_path.write_bytes(b"%PDF-test")

            documents = load_pdf(
                pdf_path,
                document_id="test-document-id",
            )

        self.assertEqual(len(documents), 2)

        # First page
        self.assertEqual(
            documents[0].page_content,
            "First page text.",
        )

        self.assertEqual(
            documents[0].metadata["source"],
            "sample.pdf",
        )

        self.assertEqual(
            documents[0].metadata["file_name"],
            "sample.pdf",
        )

        self.assertEqual(
            documents[0].metadata["page"],
            1,
        )

        self.assertEqual(
            documents[0].metadata["document_id"],
            "test-document-id",
        )

        # Second page
        self.assertEqual(
            documents[1].page_content,
            "Second page text.",
        )

        self.assertEqual(
            documents[1].metadata["source"],
            "sample.pdf",
        )

        self.assertEqual(
            documents[1].metadata["file_name"],
            "sample.pdf",
        )

        self.assertEqual(
            documents[1].metadata["page"],
            2,
        )

        self.assertEqual(
            documents[1].metadata["document_id"],
            "test-document-id",
        )


if __name__ == "__main__":
    unittest.main()