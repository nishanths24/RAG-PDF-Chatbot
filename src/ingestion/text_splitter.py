"""Deterministic text chunking for RAG ingestion."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split page-level documents into RAG-ready chunks.

    Original page and document metadata are preserved on every chunk.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[Document] = []

    for document in documents:
        split_parts = splitter.split_text(document.page_content)

        for chunk_index, chunk_text in enumerate(split_parts):
            if not chunk_text.strip():
                continue

            metadata = dict(document.metadata)

            document_id = str(metadata.get("document_id", "unknown"))
            page = metadata.get("page", "unknown")

            metadata["chunk_index"] = chunk_index
            metadata["chunk_id"] = (
                f"{document_id}-p{page}-c{chunk_index}"
            )

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata,
                )
            )

    return chunks