"""Lightweight routing layer to categorize queries."""

from __future__ import annotations

from enum import Enum
import re


class QueryMode(str, Enum):
    CONVERSATIONAL = "conversational"
    APP_HELP = "app_help"
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
            r"^(how are you)$",
            r"^(who are you|what are you|what do you do|what can you do|what is your purpose|how can you help me|how can you help|how you will help|how will you help|how you will help me|how will you help me|how do you help|how do you help me|what can you help me with|explain what you do|tell me about yourself)$",
            r"^(this is useless|you didnt answer|thats wrong|why cant you understand)$"
        ]

        self.app_help_patterns = [
            r"where should i upload",
            r"where can i upload",
            r"how do i upload",
            r"how can i upload",
            r"where do i upload",
            r"where is the upload",
            r"where is the document library",
            r"how do i remove",
            r"how can i remove",
            r"how do i delete",
            r"how many pdfs can i upload",
            r"where can i see my documents",
            r"how do i change the number of sources",
            r"what does the upload button do",
            r"how does this chatbot work",
            r"how do i use this chatbot"
        ]

    def route(self, query: str) -> QueryMode:
        """Categorize a query based on rules and heuristics."""
        query_lower = re.sub(r'[^a-z0-9\s]', '', query.strip().lower())
        query_lower = " ".join(query_lower.split())

        # Check conversational
        for pattern in self.conversational_patterns:
            if re.match(pattern, query_lower):
                return QueryMode.CONVERSATIONAL

        # Check app help
        if any(pattern in query_lower for pattern in self.app_help_patterns):
            return QueryMode.APP_HELP

        # Check synthesis
        synthesis_keywords = [
            "what is this document about",
            "summarize",
            "summary",
            "overview",
            "main points",
            "key topics",
            "main findings",
            "objectives methodology and results",
            "all pdf",
            "all document",
            "these pdf",
            "these document",
            "uploaded pdf",
            "uploaded document",
            "across the document"
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
        if mode in (QueryMode.CONVERSATIONAL, QueryMode.APP_HELP):
            return 0
        elif mode == QueryMode.DIRECT:
            return 3
        elif mode == QueryMode.EXPLANATION:
            return min(5, default_k) if default_k >= 5 else 4
        elif mode == QueryMode.SYNTHESIS:
            return max(6, default_k)
        return default_k
