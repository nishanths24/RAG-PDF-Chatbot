"""Text splitting helpers.

Phase 1 provides the module layout only. Chunking is not implemented.
"""

from __future__ import annotations

from typing import Any


def split_documents(documents: list[Any]) -> list[Any]:
    """Split documents into overlapping chunks.

    Args:
        documents: Loaded document objects.

    Returns:
        Chunked document objects.

    Raises:
        NotImplementedError: Always in Phase 1.
    """
    raise NotImplementedError("Text splitting is not implemented in Phase 1.")
