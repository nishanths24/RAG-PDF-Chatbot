"""Tests for text splitting. Chunking is not implemented in Phase 1."""

from __future__ import annotations

import unittest

from src.ingestion.text_splitter import split_documents


class TextSplitterPhase1Tests(unittest.TestCase):
    """Confirm the Phase 1 splitter contract."""

    def test_split_documents_is_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            split_documents([])


if __name__ == "__main__":
    unittest.main()
