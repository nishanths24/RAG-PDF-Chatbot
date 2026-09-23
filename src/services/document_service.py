"""Service for processing and indexing PDF documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from src.embeddings.embedding_service import EmbeddingService
from src.ingestion.document_processor import process_pdf
from src.ingestion.text_splitter import split_documents
from src.vectorstore.faiss_store import FAISSVectorStore


@dataclass(frozen=True)
class IndexingResult:
    """Summary of a PDF indexing operation."""

    file_name: str
    document_id: str
    pages: int
    chunks: int
    vectors: int


class DocumentService:
    """Coordinate PDF ingestion, chunking, embedding, and indexing."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        """Initialize the document indexing service."""

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap must be greater than or equal to zero."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def index_pdf(
        self,
        file_path: str | Path,
    ) -> IndexingResult:
        """Process and index a PDF into the FAISS vector store."""

        # ---------------------------------------------------------
        # 1. Extract PDF pages
        # ---------------------------------------------------------
        page_documents = process_pdf(file_path)

        if not page_documents:
            raise ValueError(
                "PDF produced no documents."
            )

        # ---------------------------------------------------------
        # 2. Split pages into chunks
        # ---------------------------------------------------------
        chunks = split_documents(
            page_documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        if not chunks:
            raise ValueError(
                "PDF produced no text chunks."
            )

        # ---------------------------------------------------------
        # 3. Extract text for embedding
        # ---------------------------------------------------------
        texts = [
            document.page_content
            for document in chunks
        ]

        if not texts:
            raise ValueError(
                "PDF produced no text for embedding."
            )

        # ---------------------------------------------------------
        # 4. Generate local embeddings
        # ---------------------------------------------------------
        embeddings = (
            self.embedding_service.embed_documents(
                texts
            )
        )

        if not embeddings:
            raise ValueError(
                "Embedding service produced no vectors."
            )

        # ---------------------------------------------------------
        # 5. Add vectors + metadata to FAISS
        # ---------------------------------------------------------
        self.vector_store.add_documents(
            chunks,
            embeddings,
        )

        # ---------------------------------------------------------
        # 6. Persist FAISS index
        # ---------------------------------------------------------
        self.vector_store.save()

        # ---------------------------------------------------------
        # 7. Read document metadata
        # ---------------------------------------------------------
        first_document = page_documents[0]

        document_id = str(
            first_document.metadata.get(
                "document_id",
                "",
            )
        )

        file_name = str(
            first_document.metadata.get(
                "file_name",
                Path(file_path).name,
            )
        )

        # ---------------------------------------------------------
        # 8. Return indexing summary
        # ---------------------------------------------------------
        return IndexingResult(
            file_name=file_name,
            document_id=document_id,
            pages=len(page_documents),
            chunks=len(chunks),
            vectors=len(embeddings),
        )

    def load_existing_index(self) -> None:
        """Load an existing persistent FAISS index."""

        self.vector_store.load()

    def clear_index(self) -> None:
        """Clear all indexed documents."""

        self.vector_store.clear()

    def get_indexed_document_count(self) -> int:
        """Return the number of indexed chunks."""

        return self.vector_store.count()

    def is_index_empty(self) -> bool:
        """Return whether the index contains no chunks."""

        return self.vector_store.is_empty()