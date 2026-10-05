"""Persistent document registry for multi-document RAG."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for an indexed document."""
    document_id: str
    file_name: str
    file_hash: str
    page_count: int
    chunk_count: int
    indexed_at: str


class DocumentRegistry:
    """Manages the persistent list of indexed documents."""

    def __init__(self, persist_directory: str | Path) -> None:
        self.persist_directory = Path(persist_directory)
        self.registry_path = self.persist_directory / "registry.json"
        self._documents: dict[str, DocumentMetadata] = {}

    def add_document(self, metadata: DocumentMetadata) -> None:
        """Add a document to the registry and save."""
        self._documents[metadata.document_id] = metadata
        self.save()

    def remove_document(self, document_id: str) -> None:
        """Remove a document by ID."""
        if document_id in self._documents:
            del self._documents[document_id]
            self.save()

    def get_document(self, document_id: str) -> DocumentMetadata | None:
        """Get document metadata by ID."""
        return self._documents.get(document_id)

    def get_all_documents(self) -> list[DocumentMetadata]:
        """Return all registered documents sorted by insertion/indexed time."""
        docs = list(self._documents.values())
        return sorted(docs, key=lambda d: d.indexed_at)

    def contains_hash(self, file_hash: str) -> bool:
        """Check if a file hash is already indexed."""
        return any(doc.file_hash == file_hash for doc in self._documents.values())

    def clear(self) -> None:
        """Clear all documents from the registry and remove the file."""
        self._documents.clear()
        if self.registry_path.exists():
            try:
                self.registry_path.unlink()
            except OSError as exc:
                logger.error(f"Failed to remove registry file: {exc}")

    def save(self) -> None:
        """Persist the registry to disk."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self._documents.items()}
        try:
            with self.registry_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"Failed to save registry: {exc}")

    def load(self) -> None:
        """Load the registry from disk."""
        if not self.registry_path.exists():
            self._documents = {}
            return

        try:
            with self.registry_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self._documents = {
                    k: DocumentMetadata(**v) for k, v in data.items()
                }
        except Exception as exc:
            logger.error(f"Failed to load registry: {exc}")
            self._documents = {}
