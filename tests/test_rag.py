"""Tests for the RAG generation layer."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from src.rag.ollama_service import OllamaService, OllamaServiceError
from src.rag.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.rag.qa_chain import QAChain, QAResponse
from src.retrieval.retriever import RetrievalResult


class PromptTests(unittest.TestCase):
    """Test RAG prompt construction."""

    def test_prompt_contains_question_and_context(self) -> None:
        prompt = build_rag_prompt(
            question="What is machine learning?",
            context="Machine learning is a subset of artificial intelligence.",
        )

        self.assertIn("What is machine learning?", prompt)
        self.assertIn(
            "Machine learning is a subset of artificial intelligence.",
            prompt,
        )
        self.assertIn(SYSTEM_PROMPT, prompt)

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_rag_prompt("", "Some context")

    def test_empty_context_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_rag_prompt("Some question", "")


class OllamaServiceTests(unittest.TestCase):
    """Test the local Ollama service without making real API calls."""

    def test_configuration_is_stored(self) -> None:
        service = OllamaService(
            model="qwen2.5:3b",
            base_url="http://localhost:11434/",
            timeout=60,
        )

        self.assertEqual(service.model, "qwen2.5:3b")
        self.assertEqual(service.base_url, "http://localhost:11434")
        self.assertEqual(service.timeout, 60)

    @patch("src.rag.ollama_service.requests.get")
    def test_is_available_returns_true_for_http_200(
        self,
        mock_get: Mock,
    ) -> None:
        mock_get.return_value.status_code = 200

        service = OllamaService()

        self.assertTrue(service.is_available())
        mock_get.assert_called_once()

    @patch("src.rag.ollama_service.requests.get")
    def test_is_available_returns_false_when_connection_fails(
        self,
        mock_get: Mock,
    ) -> None:
        import requests

        mock_get.side_effect = requests.RequestException("Connection failed")

        service = OllamaService()

        self.assertFalse(service.is_available())

    @patch("src.rag.ollama_service.requests.post")
    def test_generate_returns_model_response(
        self,
        mock_post: Mock,
    ) -> None:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": "Machine learning is a branch of AI."
        }

        service = OllamaService()

        result = service.generate("What is machine learning?")

        self.assertEqual(
            result,
            "Machine learning is a branch of AI.",
        )

        mock_post.assert_called_once()

    def test_empty_prompt_is_rejected(self) -> None:
        service = OllamaService()

        with self.assertRaises(ValueError):
            service.generate("")


class QAChainTests(unittest.TestCase):
    """Test retrieval-to-generation behavior."""

    def setUp(self) -> None:
        self.retriever = Mock()
        self.llm = Mock()
        self.chain = QAChain(
            retriever=self.retriever,
            llm=self.llm,
        )

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.chain.ask("")

    def test_no_retrieval_results_returns_fallback(self) -> None:
        self.retriever.retrieve.return_value = []

        result = self.chain.ask("What is machine learning?")

        self.assertIsInstance(result, QAResponse)
        self.assertEqual(result.sources, [])
        self.assertIn(
            "couldn't find relevant information",
            result.answer.lower(),
        )
        self.llm.generate.assert_not_called()

    def test_retrieved_context_is_sent_to_llm(self) -> None:
        document = Document(
            page_content="Machine learning is a subset of AI.",
            metadata={
                "file_name": "sample.pdf",
                "page": 3,
            },
        )

        retrieval_result = RetrievalResult(
            document=document,
            score=0.92,
        )

        self.retriever.retrieve.return_value = [retrieval_result]
        self.llm.generate.return_value = (
            "Machine learning is a subset of artificial intelligence."
        )

        result = self.chain.ask("What is machine learning?")

        self.assertEqual(
            result.answer,
            "Machine learning is a subset of artificial intelligence.",
        )
        self.assertEqual(result.sources, [retrieval_result])

        self.llm.generate.assert_called_once()

        generated_prompt = self.llm.generate.call_args.args[0]

        self.assertIn(
            "Machine learning is a subset of AI.",
            generated_prompt,
        )
        self.assertIn(
            "sample.pdf",
            generated_prompt,
        )
        self.assertIn(
            "Page 3",
            generated_prompt,
        )
        self.assertIn(
            "What is machine learning?",
            generated_prompt,
        )


if __name__ == "__main__":
    unittest.main()