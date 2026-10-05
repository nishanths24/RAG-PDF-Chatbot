"""Application service for RAG-powered chat."""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.qa_chain import QAChain
from src.retrieval.retriever import RetrievalResult


@dataclass(frozen=True)
class ChatResponse:
    """Response returned by the chat service."""

    answer: str
    sources: list[RetrievalResult]


class ChatService:
    """Provide a simple application interface for RAG questions."""

    def __init__(self, qa_chain: QAChain) -> None:
        self.qa_chain = qa_chain

    def ask(self, question: str, history: list[dict[str, str]] | None = None) -> ChatResponse:
        """Answer a question using the configured RAG chain."""
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question must not be empty.")

        response = self.qa_chain.ask(cleaned_question, history=history)

        return ChatResponse(
            answer=response.answer,
            sources=response.sources,
        )


def ask_question(question: str, qa_chain: QAChain, history: list[dict[str, str]] | None = None) -> str:
    """Convenience function for answering a question."""
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question must not be empty.")

    response = qa_chain.ask(cleaned_question, history=history)

    return response.answer