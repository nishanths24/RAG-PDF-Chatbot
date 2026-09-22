"""Orchestration for PDF ingestion into page-level documents."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.config.settings import get_settings
from src.ingestion.pdf_loader import load_pdf, validate_pdf_path
from src.utils.helpers import calculate_file_hash
from src.utils.logger import get_logger

logger = get_logger(__name__)


def process_pdf(file_path: str | Path) -> list[Document]:
    """Ingest a PDF into page-level LangChain Documents.

    Resolves relative paths against the configured uploads directory when
    the given path is not found. Embeddings, indexing, and retrieval are
    not performed.

    Args:
        file_path: Path to a PDF, or a filename under the uploads directory.

    Returns:
        Page-level documents ready for future chunking.

    Raises:
        src.ingestion.exceptions.IngestionError: For validation and extraction
            failures with a user-facing ``user_message``.
    """
    path = _resolve_pdf_path(file_path)
    path = validate_pdf_path(path)

    logger.info("PDF ingestion started file_name=%s", path.name)
    document_id = calculate_file_hash(path)
    logger.info("document ID generated file_name=%s document_id=%s", path.name, document_id)

    documents = load_pdf(path, document_id=document_id)
    logger.info(
        "PDF ingestion completed file_name=%s document_id=%s extracted_pages=%s",
        path.name,
        document_id,
        len(documents),
    )
    return documents


def _resolve_pdf_path(file_path: str | Path) -> Path:
    """Resolve a PDF path, falling back to the configured uploads directory."""
    path = Path(file_path)
    try:
        if path.is_file():
            return path
    except OSError:
        return path

    if not path.is_absolute():
        upload_candidate = get_settings().uploads_dir / path
        try:
            if upload_candidate.is_file():
                return upload_candidate
        except OSError:
            return path
    return path
