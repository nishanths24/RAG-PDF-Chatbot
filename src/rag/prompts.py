"""Prompt templates for the RAG pipeline."""

from __future__ import annotations


SYSTEM_PROMPT = """You are a helpful PDF question-answering assistant.

Answer questions using only the provided context from the uploaded PDF documents.

Rules:
1. Use only information supported by the provided context.
2. If the answer is not present in the context, say:
   "I couldn't find that information in the uploaded documents."
3. Do not invent facts, sources, page numbers, or quotations.
4. Keep the answer clear and concise.
5. When possible, mention the relevant page number from the context.
"""


def build_rag_prompt(question: str, context: str) -> str:
    """Build a grounded RAG prompt."""
    cleaned_question = question.strip()
    cleaned_context = context.strip()

    if not cleaned_question:
        raise ValueError("Question must not be empty.")

    if not cleaned_context:
        raise ValueError("Context must not be empty.")

    return f"""{SYSTEM_PROMPT}

CONTEXT FROM DOCUMENTS:
-----------------------
{cleaned_context}
-----------------------

QUESTION:
{cleaned_question}

ANSWER:
"""