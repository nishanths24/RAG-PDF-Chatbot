import unittest
from unittest.mock import MagicMock

from src.rag.qa_chain import QAChain
from src.retrieval.retriever import Retriever, RetrievalResult
from src.services.document_registry import DocumentRegistry, DocumentMetadata
from src.orchestration.query_router import QueryMode
from langchain_core.documents import Document

class RuntimeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.retriever = MagicMock(spec=Retriever)
        self.retriever.top_k = 3
        self.llm = MagicMock()
        self.llm.generate.return_value = "Mock answer"
        self.registry = DocumentRegistry("dummy_path")
        
        # Add two documents to match user's regression scenario
        self.registry.add_document(DocumentMetadata("doc_1", "Business.pdf", "hash1", 1, 1, "time1"))
        self.registry.add_document(DocumentMetadata("doc_2", "SM PDF 2.pdf", "hash2", 1, 1, "time2"))
        
        self.qa_chain = QAChain(self.retriever, self.llm, self.registry)

    def test_conversational_routing(self):
        # 1. "Who are you?" → sources=[]
        resp = self.qa_chain.ask("Who are you?")
        self.assertEqual(len(resp.sources), 0)
        self.retriever.retrieve.assert_not_called()

    def test_app_help_routing(self):
        # 2. "Where do I upload PDFs?" → sources=[]
        resp = self.qa_chain.ask("Where do I upload PDFs?")
        self.assertEqual(len(resp.sources), 0)
        self.retriever.retrieve.assert_not_called()

    def test_ordinal_resolution_first(self):
        # 3. "What is the first document about?" → first registry document_id
        self.retriever.retrieve.return_value = [
            RetrievalResult(Document(page_content="test", metadata={"document_id": "doc_1", "file_name": "Business.pdf", "page": 1}), score=0.9)
        ]
        resp = self.qa_chain.ask("What is the first document about?")
        self.retriever.retrieve.assert_called_with(
            "What is the first document about?",
            document_id="doc_1",
            balance_synthesis=False,
            active_document_ids=['doc_1', 'doc_2']
        )
        self.assertEqual(len(resp.sources), 1)

    def test_ordinal_resolution_second(self):
        # 4. "What is the second document about?" → second registry document_id
        self.retriever.retrieve.return_value = [
            RetrievalResult(Document(page_content="test", metadata={"document_id": "doc_2", "file_name": "SM PDF 2.pdf", "page": 1}), score=0.9)
        ]
        resp = self.qa_chain.ask("What is the second document about?")
        self.retriever.retrieve.assert_called_with(
            "What is the second document about?",
            document_id="doc_2",
            balance_synthesis=False,
            active_document_ids=['doc_1', 'doc_2']
        )

    def test_explicit_filename(self):
        # 5. "What is Business.pdf about?" → Business.pdf document_id
        self.retriever.retrieve.return_value = []
        resp = self.qa_chain.ask("What is Business.pdf about?")
        self.retriever.retrieve.assert_called_with(
            "What is Business.pdf about?",
            document_id="doc_1",
            balance_synthesis=False,
            active_document_ids=['doc_1', 'doc_2']
        )

    def test_all_document_synthesis(self):
        # 6. "Summarize all uploaded documents" → document_id=None
        self.retriever.retrieve.return_value = []
        resp = self.qa_chain.ask("Summarize all uploaded documents")
        self.retriever.retrieve.assert_called_with(
            "Summarize all uploaded documents",
            document_id=None,
            balance_synthesis=True,
            active_document_ids=['doc_1', 'doc_2']
        )

if __name__ == "__main__":
    unittest.main()
