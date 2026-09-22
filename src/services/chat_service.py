"""Chat service.

Phase 1 provides the module layout only. Question answering is not implemented.
"""

from __future__ import annotations


def ask_question(question: str) -> str:
    """Answer a question using the RAG pipeline.

    Args:
        question: User question.

    Returns:
        Model answer grounded in indexed documents.

    Raises:
        NotImplementedError: Always in Phase 1.
    """
    raise NotImplementedError("Chat service is not implemented in Phase 1.")
