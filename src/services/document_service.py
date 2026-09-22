"""Document management service.

Phase 1 provides the module layout only. Upload and indexing are not implemented.
"""

from __future__ import annotations

from pathlib import Path


def ingest_document(file_path: str | Path) -> None:
    """Ingest a PDF into the vector store.

    Args:
        file_path: Path to a PDF file.

    Raises:
        NotImplementedError: Always in Phase 1.
    """
    raise NotImplementedError("Document service is not implemented in Phase 1.")
