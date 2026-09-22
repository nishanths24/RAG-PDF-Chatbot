"""Document processing orchestration.

Phase 1 provides the module layout only. Processing is not implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def process_pdf(file_path: str | Path) -> list[Any]:
    """Load and chunk a PDF for indexing.

    Args:
        file_path: Path to a PDF file.

    Returns:
        Chunked documents ready for embedding.

    Raises:
        NotImplementedError: Always in Phase 1.
    """
    raise NotImplementedError("Document processing is not implemented in Phase 1.")
