"""Page-level PDF loading into LangChain Documents."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import EmptyFileError, PdfReadError, PdfStreamError

from src.ingestion.exceptions import (
    EmptyPdfError,
    InvalidPdfPathError,
    MalformedPdfError,
    NoExtractableTextError,
    PdfAccessError,
    PdfNotFoundError,
)
from src.utils.helpers import is_pdf_filename, normalize_extracted_text
from src.utils.logger import get_logger

logger = get_logger(__name__)

METADATA_KEYS = ("source", "file_name", "page", "document_id")


def validate_pdf_path(file_path: str | Path) -> Path:
    """Validate that ``file_path`` points to an existing PDF file.

    Args:
        file_path: Candidate PDF path.

    Returns:
        The resolved path.

    Raises:
        PdfNotFoundError: If the path does not exist.
        InvalidPdfPathError: If the path is not a file or is not a PDF.
        PdfAccessError: If the path cannot be stat'ed due to permissions.
    """
    path = Path(file_path)
    try:
        exists = path.exists()
    except OSError as exc:
        logger.error("PDF path could not be accessed file_name=%s", path.name)
        raise PdfAccessError(
            "The PDF file could not be read due to a permission error."
        ) from exc

    if not exists:
        raise PdfNotFoundError("The PDF file could not be found.")
    if not path.is_file():
        raise InvalidPdfPathError("The specified path is not a file.")
    if not is_pdf_filename(path.name):
        raise InvalidPdfPathError("Only PDF files are supported.")
    return path


def load_pdf(file_path: str | Path, *, document_id: str) -> list[Document]:
    """Extract page-level LangChain Documents from a PDF.

    Empty or whitespace-only pages are skipped. Original 1-based page
    numbers are preserved on pages that yield text.

    Args:
        file_path: Path to a PDF file.
        document_id: SHA-256 digest of the PDF file content.

    Returns:
        Page-level documents with the stable ingestion metadata contract.

    Raises:
        PdfNotFoundError: If the path does not exist.
        InvalidPdfPathError: If the path is not a PDF file.
        EmptyPdfError: If the PDF has no pages or is empty.
        NoExtractableTextError: If no page contains extractable text.
        MalformedPdfError: If the file is corrupted or not a valid PDF.
        PdfAccessError: If the file cannot be read.
    """
    path = validate_pdf_path(file_path)
    file_name = path.name
    logger.info("PDF extraction started file_name=%s", file_name)

    reader = _open_pdf(path)
    page_count = len(reader.pages)
    logger.info("PDF pages discovered file_name=%s page_count=%s", file_name, page_count)

    if page_count == 0:
        logger.error("PDF extraction failed file_name=%s reason=empty_pdf", file_name)
        raise EmptyPdfError("The PDF file is empty.")

    documents: list[Document] = []
    for index, page in enumerate(reader.pages):
        page_number = index + 1
        raw_text = _extract_page_text(page, file_name=file_name, page_number=page_number)
        normalized = normalize_extracted_text(raw_text)
        if not normalized:
            continue
        documents.append(
            Document(
                page_content=normalized,
                metadata={
                    "source": file_name,
                    "file_name": file_name,
                    "page": page_number,
                    "document_id": document_id,
                },
            )
        )

    logger.info(
        "PDF pages with extracted text file_name=%s extracted_pages=%s",
        file_name,
        len(documents),
    )
    if not documents:
        logger.error(
            "PDF extraction failed file_name=%s reason=no_extractable_text",
            file_name,
        )
        raise NoExtractableTextError("No extractable text was found in the PDF.")

    logger.info("PDF extraction completed file_name=%s", file_name)
    return documents


def _open_pdf(path: Path) -> PdfReader:
    """Open a PDF with mapped, user-facing errors."""
    try:
        reader = PdfReader(str(path))
    except EmptyFileError as exc:
        logger.error("PDF extraction failed file_name=%s reason=empty_file", path.name)
        raise EmptyPdfError("The PDF file is empty.") from exc
    except PermissionError as exc:
        logger.error("PDF extraction failed file_name=%s reason=permission", path.name)
        raise PdfAccessError(
            "The PDF file could not be read due to a permission error."
        ) from exc
    except (PdfReadError, PdfStreamError) as exc:
        logger.error("PDF extraction failed file_name=%s reason=malformed", path.name)
        raise MalformedPdfError("The PDF file is invalid or corrupted.") from exc
    except OSError as exc:
        logger.error("PDF extraction failed file_name=%s reason=os_error", path.name)
        raise PdfAccessError(
            "The PDF file could not be read due to a permission error."
        ) from exc

    if reader.is_encrypted:
        logger.error("PDF extraction failed file_name=%s reason=encrypted", path.name)
        raise MalformedPdfError("The PDF file is invalid or corrupted.")

    return reader


def _extract_page_text(page: object, *, file_name: str, page_number: int) -> str | None:
    """Extract text from one page, treating missing text as empty."""
    extract_text = getattr(page, "extract_text", None)
    if extract_text is None:
        return None
    try:
        text = extract_text()
    except Exception:
        logger.error(
            "PDF page extraction failed file_name=%s page=%s",
            file_name,
            page_number,
        )
        raise
    if text is None:
        return None
    return str(text)
