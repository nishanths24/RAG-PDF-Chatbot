"""Project-specific exceptions for PDF ingestion."""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for PDF ingestion failures.

    ``user_message`` is a short, user-facing explanation without a stack trace.
    """

    def __init__(self, user_message: str) -> None:
        self.user_message = user_message
        super().__init__(user_message)


class PdfNotFoundError(IngestionError):
    """Raised when the PDF path does not exist."""


class InvalidPdfPathError(IngestionError):
    """Raised when the path exists but is not a readable PDF file."""


class EmptyPdfError(IngestionError):
    """Raised when the PDF has no pages or is an empty file."""


class NoExtractableTextError(IngestionError):
    """Raised when no page yields meaningful extracted text."""


class MalformedPdfError(IngestionError):
    """Raised when the file is not a valid PDF."""


class PdfAccessError(IngestionError):
    """Raised when the PDF cannot be read due to permissions or I/O errors."""
