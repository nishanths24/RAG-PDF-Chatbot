"""Small path and string helpers used across the project."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The same path after ensuring it exists.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_pdf_filename(filename: str) -> bool:
    """Return True when a filename has a `.pdf` extension.

    Args:
        filename: Original or uploaded file name.
    """
    return Path(filename).suffix.lower() == ".pdf"
