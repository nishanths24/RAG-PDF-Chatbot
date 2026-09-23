"""RAG question-answering chain."""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.ollama_service import OllamaService
from src.rag.prompts import build_rag_prompt
from src.retrieval.retriever import RetrievalResult, Retriever


@dataclass(frozen=True)
class QAResponse:
    """Final RAG response with retrieved sources."""

    answer: str
    sources: list[RetrievalResult]


class QAChain:
    """Combine retrieval, prompting, and local LLM generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm: OllamaService,
    ) -> None:
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str) -> QAResponse:
        """Retrieve relevant context and generate a grounded answer."""
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question must not be empty.")

        results = self.retriever.retrieve(cleaned_question)

        if not results:
            return QAResponse(
                answer=(
                    "I couldn't find relevant information in the "
                    "uploaded documents."
                ),
                sources=[],
            )

        context_parts: list[str] = []

        for index, result in enumerate(results, start=1):
            document = result.document
            metadata = document.metadata or {}

            source = metadata.get("file_name", "Unknown document")
            page = metadata.get("page", "Unknown page")

            context_parts.append(
                f"[Source {index}: {source}, Page {page}]\n"
                f"{document.page_content}"
            )

        context = "\n\n".join(context_parts)

        prompt = build_rag_prompt(
            question=cleaned_question,
            context=context,
        )

        answer = self.llm.generate(prompt)

        return QAResponse(
            answer=answer,
            sources=results,
        )