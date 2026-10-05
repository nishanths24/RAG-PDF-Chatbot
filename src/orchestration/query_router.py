"""Lightweight routing layer to categorize queries."""

from __future__ import annotations

from enum import Enum
import re


class QueryMode(str, Enum):
    CONVERSATIONAL = "conversational"
    DIRECT = "direct"
    EXPLANATION = "explanation"
    SYNTHESIS = "synthesis"


class QueryRouter:
    """Routes queries to the appropriate RAG logic without using an LLM."""

    def __init__(self) -> None:
        self.conversational_patterns = [
            r"^(hi|hello|hey|hey there|hi there)$",
            r"^(good morning|good afternoon|good evening)$",
            r"^(thanks|thank you|thank you so much)$",
            r"^(okay|ok|bye|goodbye)$",
            r"^(how are you\??)$",
            r"^(this is useless|you didn't answer|that's wrong|why can't you understand\??)$"
        ]

    def route(self, query: str) -> QueryMode:
        """Categorize a query based on rules and heuristics."""
        query_lower = query.strip().lower()

        # Check conversational
        for pattern in self.conversational_patterns:
            if re.match(pattern, query_lower):
                return QueryMode.CONVERSATIONAL

        # Check synthesis
        synthesis_keywords = [
            "what is this document about",
            "summarize",
            "summary",
            "overview",
            "main points",
            "key topics",
            "main findings",
            "objectives, methodology and results"
        ]
        if any(keyword in query_lower for keyword in synthesis_keywords):
            return QueryMode.SYNTHESIS

        # Check explanation
        explanation_keywords = [
            "explain",
            "how does",
            "why is",
            "how to"
        ]
        if any(keyword in query_lower for keyword in explanation_keywords):
            return QueryMode.EXPLANATION

        # Default to direct
        return QueryMode.DIRECT

    def get_retrieval_depth(self, mode: QueryMode, default_k: int = 3) -> int:
        """Return recommended retrieval depth based on query mode."""
        if mode == QueryMode.CONVERSATIONAL:
            return 0
        elif mode == QueryMode.DIRECT:
            return 3
        elif mode == QueryMode.EXPLANATION:
            return min(5, default_k) if default_k >= 5 else 4
        elif mode == QueryMode.SYNTHESIS:
            return max(6, default_k)
        return default_k
