"""Small path, hashing, and text helpers used across the project."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_HASH_CHUNK_SIZE = 1024 * 1024
_MULTI_SPACE = re.compile(r"[^\S\n\r]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


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
    """Return True when a filename has a ``.pdf`` extension.

    Args:
        filename: Original or uploaded file name.
    """
    return Path(filename).suffix.lower() == ".pdf"


def calculate_file_hash(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes.

    The file is read in binary chunks so large PDFs do not need to fit
    in memory. The digest depends only on file content.

    Args:
        path: Path to the file to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If the path does not exist.
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_extracted_text(text: str | None) -> str:
    """Clean PDF extraction artifacts without rewriting meaning.

    Handles ``None``, mixed newlines, non-breaking spaces, repeated spaces,
    and excessive blank lines. Punctuation, numbers, and headings are kept.

    Args:
        text: Raw page text from the PDF extractor.

    Returns:
        Deterministically normalized text, or an empty string.
    """
    if text is None:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    cleaned = cleaned.replace("\u00a0", " ").replace("\t", " ")
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    lines = [line.strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = _MULTI_NEWLINE.sub("\n\n", cleaned)
    return cleaned.strip()
