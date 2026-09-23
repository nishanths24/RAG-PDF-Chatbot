"""Application service for RAG-powered chat."""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.qa_chain import QAChain, QAResponse


@dataclass(frozen=True)
class ChatResponse:
    """Response returned by the chat service."""

    answer: str
    sources: QAResponse


class ChatService:
    """Provide a simple application interface for RAG questions."""

    def __init__(self, qa_chain: QAChain) -> None:
        self.qa_chain = qa_chain

    def ask(self, question: str) -> ChatResponse:
        """Answer a question using the configured RAG chain."""
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question must not be empty.")

        response = self.qa_chain.ask(cleaned_question)

        return ChatResponse(
            answer=response.answer,
            sources=response,
        )


def ask_question(question: str, qa_chain: QAChain) -> str:
    """Convenience function for answering a question."""
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question must not be empty.")

    response = qa_chain.ask(cleaned_question)

    return response.answer