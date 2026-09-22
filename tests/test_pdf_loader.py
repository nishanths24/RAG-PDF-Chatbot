"""Tests for PDF loading. Ingestion is not implemented in Phase 1."""

from __future__ import annotations

import unittest

from src.ingestion.pdf_loader import load_pdf


class PdfLoaderPhase1Tests(unittest.TestCase):
    """Confirm the Phase 1 loader contract."""

    def test_load_pdf_is_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            load_pdf("sample.pdf")


if __name__ == "__main__":
    unittest.main()
