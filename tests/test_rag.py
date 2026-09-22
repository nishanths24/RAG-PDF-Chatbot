"""Tests for the RAG QA chain. The pipeline is not implemented in Phase 1."""

from __future__ import annotations

import unittest

from src.rag.qa_chain import build_qa_chain


class RagPhase1Tests(unittest.TestCase):
    """Confirm the Phase 1 RAG contract."""

    def test_build_qa_chain_is_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            build_qa_chain()


if __name__ == "__main__":
    unittest.main()
