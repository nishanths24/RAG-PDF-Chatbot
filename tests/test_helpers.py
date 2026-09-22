"""Tests for shared helper functions."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.utils.helpers import calculate_file_hash, normalize_extracted_text


class HelperTests(unittest.TestCase):
    """Validate hashing and deterministic text normalization."""

    def test_calculate_file_hash_is_sha256(self) -> None:
        content = b"RAG PDF Chatbot test content"

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sample.pdf"
            file_path.write_bytes(content)

            expected_hash = hashlib.sha256(content).hexdigest()

            self.assertEqual(
                calculate_file_hash(file_path),
                expected_hash,
            )

    def test_same_content_produces_same_hash(self) -> None:
        content = b"same PDF content"

        with tempfile.TemporaryDirectory() as temp_dir:
            file_a = Path(temp_dir) / "a.pdf"
            file_b = Path(temp_dir) / "b.pdf"

            file_a.write_bytes(content)
            file_b.write_bytes(content)

            self.assertEqual(
                calculate_file_hash(file_a),
                calculate_file_hash(file_b),
            )

    def test_different_content_produces_different_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_a = Path(temp_dir) / "a.pdf"
            file_b = Path(temp_dir) / "b.pdf"

            file_a.write_bytes(b"content A")
            file_b.write_bytes(b"content B")

            self.assertNotEqual(
                calculate_file_hash(file_a),
                calculate_file_hash(file_b),
            )

    def test_text_normalization(self) -> None:
        raw_text = "  First   line.\r\n\r\n\r\n Second\tline. \r\n"

        normalized = normalize_extracted_text(raw_text)

        self.assertEqual(
            normalized,
            "First line.\n\nSecond line.",
        )

    def test_none_text_normalizes_to_empty_string(self) -> None:
        self.assertEqual(
            normalize_extracted_text(None),
            "",
        )


if __name__ == "__main__":
    unittest.main()