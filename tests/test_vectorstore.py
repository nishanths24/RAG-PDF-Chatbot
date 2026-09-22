"""Tests for the FAISS vector store. Persistence is not implemented in Phase 1."""

from __future__ import annotations

import unittest

from src.vectorstore.faiss_store import load_vectorstore, save_vectorstore


class VectorStorePhase1Tests(unittest.TestCase):
    """Confirm the Phase 1 vector store contract."""

    def test_load_vectorstore_is_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            load_vectorstore()

    def test_save_vectorstore_is_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            save_vectorstore(None)


if __name__ == "__main__":
    unittest.main()
