"""Tests for the chat service."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.rag.qa_chain import QAResponse
from src.retrieval.retriever import RetrievalResult
from src.services.chat_service import (
    ChatResponse,
    ChatService,
    ask_question,
)


class ChatServiceTests(unittest.TestCase):
    """Test application-level chat behavior."""

    def setUp(self) -> None:
        self.qa_chain = Mock()
        self.service = ChatService(self.qa_chain)

    def test_ask_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            self.service.ask("")

        self.qa_chain.ask.assert_not_called()

    def test_ask_delegates_to_qa_chain(self) -> None:
        qa_response = QAResponse(
            answer="Machine learning is a subset of AI.",
            sources=[],
        )

        self.qa_chain.ask.return_value = qa_response

        result = self.service.ask("  What is machine learning?  ")

        self.assertIsInstance(result, ChatResponse)
        self.assertEqual(
            result.answer,
            "Machine learning is a subset of AI.",
        )
        self.assertEqual(result.sources, qa_response.sources)

        self.qa_chain.ask.assert_called_once_with(
            "What is machine learning?"
        )

    def test_ask_question_returns_answer(self) -> None:
        self.qa_chain.ask.return_value = QAResponse(
            answer="This is the answer.",
            sources=[],
        )

        result = ask_question(
            "What is RAG?",
            self.qa_chain,
        )

        self.assertEqual(result, "This is the answer.")

    def test_ask_question_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            ask_question("", self.qa_chain)

        self.qa_chain.ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()