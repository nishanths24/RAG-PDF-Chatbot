"""RAG question-answering chain."""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.groq_service import GroqService
from src.rag.prompts import build_rag_prompt, SYSTEM_PROMPT
from src.retrieval.retriever import RetrievalResult, Retriever
from src.orchestration.query_router import QueryRouter, QueryMode
from src.services.document_registry import DocumentRegistry
import re


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
        document_registry: DocumentRegistry | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.router = QueryRouter()
        self.document_registry = document_registry

    def ask(self, question: str) -> QAResponse:
        """Retrieve relevant context and generate a grounded answer."""
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question must not be empty.")

        mode = self.router.route(cleaned_question)
        depth = self.router.get_retrieval_depth(mode, self.retriever.top_k)

        import logging
        logger = logging.getLogger(__name__)
        
        target_document_id = None
        if self.document_registry:
            query_lower = cleaned_question.lower()
            # Ensure we fetch latest from registry
            docs = self.document_registry.get_all_documents()
            
            logger.info(f"[QA] registry_docs={len(docs)}")
            logger.info(f"[QA] registry_names={[d.file_name for d in docs]}")

            # 1. Check ordinal references
            ordinal_map = {
                "first": 0, "1st": 0,
                "second": 1, "2nd": 1,
                "third": 2, "3rd": 2,
                "fourth": 3, "4th": 3,
                "fifth": 4, "5th": 4
            }
            
            # Match ordinal + document/pdf
            for prefix, index in ordinal_map.items():
                pattern = r'\b' + prefix + r'\b.*?\b(document|pdf)\b'
                if re.search(pattern, query_lower):
                    if index < len(docs):
                        target_document_id = docs[index].document_id
                    break
            
            # 2. Check exact filename
            if not target_document_id:
                # Sort by length descending to match longest possible filename first
                sorted_docs = sorted(docs, key=lambda d: len(d.file_name), reverse=True)
                for doc in sorted_docs:
                    fname = doc.file_name.lower()
                    fname_no_ext = fname.replace('.pdf', '')
                    
                    # Use word boundaries for safety against substring collisions (e.g. "us" in "business")
                    pattern = r'\b' + re.escape(fname_no_ext) + r'\b'
                    if re.search(pattern, query_lower) or fname in query_lower:
                        target_document_id = doc.document_id
                        break
        
        logger.info(f"[QA] query={cleaned_question}")
        logger.info(f"[QA] normalized_query={cleaned_question.lower()}")
        logger.info(f"[QA] mode={mode.value}")
        logger.info(f"[QA] resolved_document_id={target_document_id}")
        logger.info(f"[QA] retrieval_depth={depth}")

        if mode in (QueryMode.CONVERSATIONAL, QueryMode.APP_HELP):
            logger.info(f"[QA] retriever_called=False")
            
            if mode == QueryMode.CONVERSATIONAL:
                prompt = f"{SYSTEM_PROMPT}\n\nUser: {cleaned_question}\n\nAssistant:"
            else:
                prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    "The user is asking for help using this application. "
                    "This is a Streamlit PDF Chatbot. Users can upload multiple PDFs using the "
                    "Upload button in the 'Document Library' on the left sidebar. "
                    "They can view uploaded documents there, remove individual documents, "
                    "or clear the entire knowledge base. "
                    "They can also adjust the 'Retrieval Sources (top_k)' using a slider. "
                    "Please answer their question clearly and directly based on this information.\n\n"
                    f"User: {cleaned_question}\n\nAssistant:"
                )
            answer = self.llm.generate(prompt)
            return QAResponse(answer=answer, sources=[])
        
        logger.info(f"[QA] retriever_called=True")
        results = []
        original_top_k = self.retriever.top_k
        self.retriever.top_k = depth
        try:
            results = self.retriever.retrieve(
                cleaned_question,
                document_id=target_document_id,
                balance_synthesis=(mode == QueryMode.SYNTHESIS and not target_document_id)
            )
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

        # Deduplicate display sources based on document and page
        display_sources = []
        seen_display = set()
        
        for index, result in enumerate(results, start=1):
            document = result.document
            metadata = document.metadata or {}

            source = metadata.get("file_name", "Unknown document")
            page = metadata.get("page", "Unknown page")

            # Unique key for display deduplication
            disp_key = (source, page)
            if disp_key not in seen_display:
                seen_display.add(disp_key)
                display_sources.append(result)

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

        # Defensively strip any leaked [Source N] markers from the LLM answer
        # Match [Source 1], [Source 12], etc. optionally with leading whitespace
        answer = re.sub(r'\s*\[Source \d+\]', '', answer)

        return QAResponse(
            answer=answer,
            sources=display_sources,
        )