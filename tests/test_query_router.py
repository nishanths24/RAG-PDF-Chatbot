import unittest

from src.orchestration.query_router import QueryRouter, QueryMode

class TestQueryRouter(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter()

    def test_conversational_queries(self):
        queries = ["Hi", "hello", "good morning", "thanks", "how are you?"]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(self.router.route(q), QueryMode.CONVERSATIONAL)

    def test_how_you_will_help_is_conversational(self):
        queries = [
            "How you will help?",
            "How will you help?",
            "How you will help me?",
            "How can you help?",
            "How can you help me?",
            "What can you help me with?"
        ]
        for q in queries:
            with self.subTest(query=q):
                result = self.router.route(q)
                self.assertEqual(result, QueryMode.CONVERSATIONAL)
                self.assertEqual(self.router.get_retrieval_depth(result), 0)

    def test_direct_queries(self):
        queries = ["What is the main objective?", "Who is the author?"]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(self.router.route(q), QueryMode.DIRECT)

    def test_explanation_queries(self):
        queries = ["Explain the methodology.", "How does this work?"]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(self.router.route(q), QueryMode.EXPLANATION)

    def test_synthesis_queries(self):
        queries = ["What is this document about?", "Summarize the text", "Summarize all PDFs"]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(self.router.route(q), QueryMode.SYNTHESIS)

    def test_app_help_queries(self):
        queries = [
            "Where should I upload pdfs",
            "How do I upload a PDF?",
            "Where is the document library?",
            "How do I remove a document?",
            "where can i see my documents?"
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(self.router.route(q), QueryMode.APP_HELP)

    def test_boundaries(self):
        # Must be DOCUMENT/RAG (not APP_HELP or CONVERSATIONAL)
        queries = [
            "what is this document about",
            "what is this pdf about",
            "what does this document do",
            "what does the project do",
            "what is the purpose of this document",
            "explain this document",
            "summarize this pdf",
            "what are the main findings",
            "what are the key topics",
            "what all pdfs contain",
            "summarize all documents"
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertNotIn(self.router.route(q), (QueryMode.APP_HELP, QueryMode.CONVERSATIONAL))

    def test_retrieval_depth(self):
        self.assertEqual(self.router.get_retrieval_depth(QueryMode.CONVERSATIONAL), 0)
        self.assertEqual(self.router.get_retrieval_depth(QueryMode.APP_HELP), 0)
        self.assertEqual(self.router.get_retrieval_depth(QueryMode.DIRECT), 3)
        self.assertTrue(self.router.get_retrieval_depth(QueryMode.EXPLANATION, default_k=6) >= 4)
        self.assertTrue(self.router.get_retrieval_depth(QueryMode.SYNTHESIS, default_k=3) >= 6)

if __name__ == "__main__":
    unittest.main()
