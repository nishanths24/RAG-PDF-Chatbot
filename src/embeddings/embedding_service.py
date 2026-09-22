"""Embedding service.

Phase 1 provides the module layout only. Embeddings are not implemented.
"""

from __future__ import annotations

from typing import Any


def get_embedding_model() -> Any:
    """Return the configured embedding model.

    Raises:
        NotImplementedError: Always in Phase 1.
    """
    raise NotImplementedError("Embedding service is not implemented in Phase 1.")
