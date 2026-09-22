"""PDF loading helpers.

Phase 1 provides the module layout only. Document loading is not implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_pdf(file_path: str | Path) -> list[Any]:
    """Load a PDF file into document objects.

    Args:
        file_path: Path to a PDF file.

    Returns:
        A list of loaded documents.

    Raises:
        NotImplementedError: Always in Phase 1.
    """
    raise NotImplementedError("PDF loading is not implemented in Phase 1.")
