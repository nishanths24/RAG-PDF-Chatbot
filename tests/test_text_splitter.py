"""Tests for deterministic RAG document chunking."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from src.ingestion.text_splitter import split_documents


class TextSplitterTests(unittest.TestCase):
    """Validate chunking behavior and metadata preservation."""

    def create_document(
        self,
        text: str,
        *,
        page: int = 1,
        document_id: str = "document-123",
    ) -> Document:
        return Document(
            page_content=text,
            metadata={
                "source": "sample.pdf",
                "file_name": "sample.pdf",
                "page": page,
                "document_id": document_id,
            },
        )

    def test_empty_input_returns_empty_list(self) -> None:
        result = split_documents([])

        self.assertEqual(result, [])

    def test_long_document_is_split(self) -> None:
        text = " ".join(f"word{i}" for i in range(200))
        document = self.create_document(text)

        chunks = split_documents(
            [document],
            chunk_size=100,
            chunk_overlap=20,
        )

        self.assertGreater(len(chunks), 1)

    def test_chunk_size_is_respected(self) -> None:
        text = " ".join(f"word{i}" for i in range(100))
        document = self.create_document(text)

        chunks = split_documents(
            [document],
            chunk_size=100,
            chunk_overlap=20,
        )

        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 100)

    def test_metadata_is_preserved(self) -> None:
        text = " ".join(f"word{i}" for i in range(100))
        document = self.create_document(
            text,
            page=5,
            document_id="abc123",
        )

        chunks = split_documents(
            [document],
            chunk_size=100,
            chunk_overlap=20,
        )

        self.assertGreater(len(chunks), 0)

        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "sample.pdf")
            self.assertEqual(chunk.metadata["file_name"], "sample.pdf")
            self.assertEqual(chunk.metadata["page"], 5)
            self.assertEqual(chunk.metadata["document_id"], "abc123")

    def test_chunk_metadata_is_added(self) -> None:
        text = " ".join(f"word{i}" for i in range(100))
        document = self.create_document(
            text,
            page=2,
            document_id="abc123",
        )

        chunks = split_documents(
            [document],
            chunk_size=100,
            chunk_overlap=20,
        )

        self.assertGreater(len(chunks), 0)

        chunk_ids = set()

        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata["chunk_index"], index)
            self.assertIn("chunk_id", chunk.metadata)

            expected_id = f"abc123-p2-c{index}"
            self.assertEqual(chunk.metadata["chunk_id"], expected_id)

            chunk_ids.add(chunk.metadata["chunk_id"])

        self.assertEqual(len(chunk_ids), len(chunks))

    def test_multiple_documents_keep_their_metadata(self) -> None:
        document_one = self.create_document(
            " ".join(f"first{i}" for i in range(100)),
            page=1,
            document_id="document-one",
        )

        document_two = self.create_document(
            " ".join(f"second{i}" for i in range(100)),
            page=2,
            document_id="document-two",
        )

        chunks = split_documents(
            [document_one, document_two],
            chunk_size=100,
            chunk_overlap=20,
        )

        first_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata["document_id"] == "document-one"
        ]

        second_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata["document_id"] == "document-two"
        ]

        self.assertGreater(len(first_chunks), 0)
        self.assertGreater(len(second_chunks), 0)

        for chunk in first_chunks:
            self.assertEqual(chunk.metadata["page"], 1)

        for chunk in second_chunks:
            self.assertEqual(chunk.metadata["page"], 2)

    def test_input_documents_are_not_modified(self) -> None:
        document = self.create_document(
            " ".join(f"word{i}" for i in range(100))
        )

        original_text = document.page_content
        original_metadata = dict(document.metadata)

        split_documents(
            [document],
            chunk_size=100,
            chunk_overlap=20,
        )

        self.assertEqual(document.page_content, original_text)
        self.assertEqual(document.metadata, original_metadata)

    def test_invalid_chunk_size_raises_error(self) -> None:
        document = self.create_document("some text")

        with self.assertRaises(ValueError):
            split_documents(
                [document],
                chunk_size=0,
            )

        with self.assertRaises(ValueError):
            split_documents(
                [document],
                chunk_size=-1,
            )

    def test_invalid_chunk_overlap_raises_error(self) -> None:
        document = self.create_document("some text")

        with self.assertRaises(ValueError):
            split_documents(
                [document],
                chunk_size=100,
                chunk_overlap=-1,
            )

        with self.assertRaises(ValueError):
            split_documents(
                [document],
                chunk_size=100,
                chunk_overlap=100,
            )

        with self.assertRaises(ValueError):
            split_documents(
                [document],
                chunk_size=100,
                chunk_overlap=150,
            )


if __name__ == "__main__":
    unittest.main()