"""Text splitting helpers.

Chunking is implemented in a later phase. The signature accepts the
page-level LangChain Documents produced by the ingestion layer.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document


def split_documents(documents: Sequence[Document]) -> list[Document]:
    """Split page-level documents into overlapping chunks.

    Args:
        documents: Page-level documents from PDF ingestion.

    Returns:
        Chunked documents ready for embedding.

    Raises:
        NotImplementedError: Chunking is not implemented in this phase.
    """
    raise NotImplementedError("Text splitting is not implemented yet.")
