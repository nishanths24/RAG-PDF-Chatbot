"""RAG question-answering chain."""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.groq_service import GroqService
from src.rag.prompts import build_rag_prompt
from src.retrieval.retriever import RetrievalResult, Retriever
from src.orchestration.query_router import QueryRouter, QueryMode


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
        llm: GroqService,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.router = QueryRouter()

    def ask(self, question: str) -> QAResponse:
        """Retrieve relevant context and generate a grounded answer."""
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question must not be empty.")

        mode = self.router.route(cleaned_question)
        depth = self.router.get_retrieval_depth(mode, self.retriever.top_k)
        
        results = []
        if mode != QueryMode.CONVERSATIONAL:
            original_top_k = self.retriever.top_k
            self.retriever.top_k = depth
            try:
                results = self.retriever.retrieve(cleaned_question)
            finally:
                self.retriever.top_k = original_top_k

            if not results:
                return QAResponse(
                    answer=(
                        "I couldn't find enough information in the "
                        "uploaded document to answer that confidently."
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

        context = "\n\n".join(context_parts) if context_parts else ""

        prompt = build_rag_prompt(
            question=cleaned_question,
            context=context,
        )

        answer = self.llm.generate(prompt)

        return QAResponse(
            answer=answer,
            sources=results,
        )