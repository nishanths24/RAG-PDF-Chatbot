"""FAISS vector store helpers.

Phase 1 provides the module layout only. Persistence is not implemented.
"""

from __future__ import annotations

from typing import Any


def load_vectorstore() -> Any:
    """Load a persisted FAISS index.

    Raises:
        NotImplementedError: Always in Phase 1.
    """
    raise NotImplementedError("FAISS vector store is not implemented in Phase 1.")


def save_vectorstore(store: Any) -> None:
    """Persist a FAISS index to disk.

    Args:
        store: Vector store instance.

    Raises:
        NotImplementedError: Always in Phase 1.
    """
    raise NotImplementedError("FAISS vector store is not implemented in Phase 1.")
