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
        queries = ["What is this document about?", "Summarize the text"]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(self.router.route(q), QueryMode.SYNTHESIS)

    def test_retrieval_depth(self):
        self.assertEqual(self.router.get_retrieval_depth(QueryMode.CONVERSATIONAL), 0)
        self.assertEqual(self.router.get_retrieval_depth(QueryMode.DIRECT), 3)
        self.assertTrue(self.router.get_retrieval_depth(QueryMode.EXPLANATION, default_k=6) >= 4)
        self.assertTrue(self.router.get_retrieval_depth(QueryMode.SYNTHESIS, default_k=3) >= 6)

if __name__ == "__main__":
    unittest.main()
