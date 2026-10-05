import unittest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.rag.qa_chain import QAChain
from src.retrieval.retriever import Retriever, RetrievalResult
from src.services.document_registry import DocumentRegistry, DocumentMetadata
from src.rag.groq_service import GroqService


class DocumentResolutionTests(unittest.TestCase):
    def setUp(self):
        self.retriever = MagicMock(spec=Retriever)
        self.retriever.top_k = 3
        
        self.llm = MagicMock(spec=GroqService)
        self.llm.generate.return_value = "Mock response"
        
        self.registry = DocumentRegistry(persist_directory="dummy_dir")
        
        # Add test documents
        self.doc1 = DocumentMetadata(
            document_id="doc_id_1",
            file_name="Business.pdf",
            file_hash="hash1",
            page_count=10,
            chunk_count=20,
            indexed_at="time"
        )
        self.doc2 = DocumentMetadata(
            document_id="doc_id_2",
            file_name="SM PDF 2.pdf",
            file_hash="hash2",
            page_count=5,
            chunk_count=10,
            indexed_at="time"
        )
        self.doc3 = DocumentMetadata(
            document_id="doc_id_3",
            file_name="1AM22CI062_SWOT_Analysis_Report.pdf",
            file_hash="hash3",
            page_count=2,
            chunk_count=4,
            indexed_at="time"
        )
        
        self.registry.add_document(self.doc1)
        self.registry.add_document(self.doc2)
        self.registry.add_document(self.doc3)
        
        self.qa_chain = QAChain(
            retriever=self.retriever,
            llm=self.llm,
            document_registry=self.registry
        )

    def test_how_you_will_help_never_retrieves(self):
        self.retriever.retrieve.reset_mock()
        resp = self.qa_chain.ask("How you will help?")
        self.assertEqual(len(resp.sources), 0)
        self.retriever.retrieve.assert_not_called()

    def test_who_are_you_never_retrieves(self):
        self.retriever.retrieve.reset_mock()
        resp = self.qa_chain.ask("Who are you?")
        self.assertEqual(len(resp.sources), 0)
        self.retriever.retrieve.assert_not_called()

    def test_app_help_never_retrieves(self):
        self.retriever.retrieve.reset_mock()
        resp = self.qa_chain.ask("Where should I upload PDFs?")
        self.assertEqual(len(resp.sources), 0)
        self.retriever.retrieve.assert_not_called()

    def test_first_document_targets_registry_first(self):
        self.retriever.retrieve.return_value = []
        self.qa_chain.ask("What is the first document about?")
        self.retriever.retrieve.assert_called_with("What is the first document about?", document_id="doc_id_1", balance_synthesis=False)

    def test_second_document_targets_registry_second(self):
        self.retriever.retrieve.return_value = []
        self.qa_chain.ask("What is the second document about?")
        self.retriever.retrieve.assert_called_with("What is the second document about?", document_id="doc_id_2", balance_synthesis=False)

    def test_ordinal_resolution_after_multiple_uploads(self):
        # Simulate state sharing / cached service instance
        self.retriever.retrieve.return_value = []
        self.qa_chain.ask("What is the first document about?")
        self.retriever.retrieve.assert_called_with("What is the first document about?", document_id="doc_id_1", balance_synthesis=False)
        
        # Add a new doc
        self.doc4 = DocumentMetadata("doc_id_4", "New.pdf", "hash4", 1, 1, "time4")
        self.registry.add_document(self.doc4)
        
        self.retriever.retrieve.reset_mock()
        self.qa_chain.ask("What is the fourth document about?")
        self.retriever.retrieve.assert_called_with("What is the fourth document about?", document_id="doc_id_4", balance_synthesis=False)

    def test_synthesis_passes_document_id_none(self):
        self.retriever.retrieve.return_value = []
        self.qa_chain.ask("What all PDFs contain?")
        self.retriever.retrieve.assert_called_with("What all PDFs contain?", document_id=None, balance_synthesis=True)

    def test_synthesis_can_retrieve_from_multiple_documents(self):
        # We need to simulate the retriever actually returning mixed documents.
        # When document_id=None and balance_synthesis=True, it should retrieve broad.
        self.retriever.retrieve.return_value = [
            RetrievalResult(Document(page_content="biz", metadata={"document_id": "doc_id_1", "file_name": "biz.pdf", "page": 1}), score=0.9),
            RetrievalResult(Document(page_content="sm", metadata={"document_id": "doc_id_2", "file_name": "sm.pdf", "page": 1}), score=0.8)
        ]
        resp = self.qa_chain.ask("What all PDFs contain?")
        # Display sources should deduplicate the first two which are on same file and page
        self.assertEqual(len(resp.sources), 2)
        # Ensure that both documents appear in the context passed to the LLM
        context_passed = self.llm.generate.call_args[0][0]
        self.assertIn("biz", context_passed)
        self.assertIn("sm", context_passed)

    def test_clean_internal_citations_removes_markers(self):
        # Mock LLM to return a string with leaked citations
        self.llm.generate.return_value = "MainActivity is used to configure the UI. [Source 1]"
        self.retriever.retrieve.return_value = [
            RetrievalResult(Document(page_content="text", metadata={"document_id": "doc_id_1"}), score=0.8)
        ]
        resp = self.qa_chain.ask("What is MainActivity?")
        self.assertEqual(resp.answer, "MainActivity is used to configure the UI.")

    def test_clean_internal_citations_preserves_legitimate_brackets(self):
        self.llm.generate.return_value = "Section [1] discusses trading."
        self.retriever.retrieve.return_value = [
            RetrievalResult(Document(page_content="text", metadata={"document_id": "doc_id_1"}), score=0.8)
        ]
        resp = self.qa_chain.ask("What is trading?")
        self.assertEqual(resp.answer, "Section [1] discusses trading.")

    def test_source_display_deduplicates_same_page(self):
        self.retriever.retrieve.return_value = [
            RetrievalResult(Document(page_content="text1", metadata={"file_name": "A.pdf", "page": 1}), score=0.8),
            RetrievalResult(Document(page_content="text2", metadata={"file_name": "A.pdf", "page": 1}), score=0.7),
            RetrievalResult(Document(page_content="text3", metadata={"file_name": "A.pdf", "page": 2}), score=0.6),
        ]
        resp = self.qa_chain.ask("What is this document about?")
        # Display sources should deduplicate the first two which are on same file and page
        self.assertEqual(len(resp.sources), 2)
        self.assertEqual(resp.sources[0].document.page_content, "text1")
        self.assertEqual(resp.sources[1].document.page_content, "text3")

    def test_filename_query_filters_to_exact_document(self):
        self.retriever.retrieve.return_value = []
        self.qa_chain.ask("What is Business.pdf about?")
        self.retriever.retrieve.assert_called_with("What is Business.pdf about?", document_id="doc_id_1", balance_synthesis=False)

    def test_registry_updates_are_visible_to_cached_services(self):
        # Already tested by test_ordinal_resolution_after_multiple_uploads
        pass

    def test_conversational_sources_are_empty(self):
        self.retriever.retrieve.reset_mock()
        resp = self.qa_chain.ask("hi there")
        self.assertEqual(len(resp.sources), 0)

    def test_app_help_sources_are_empty(self):
        self.retriever.retrieve.reset_mock()
        resp = self.qa_chain.ask("how do i upload")
        self.assertEqual(len(resp.sources), 0)

    def test_rag_queries_still_retrieve(self):
        self.retriever.retrieve.return_value = []
        self.qa_chain.ask("What is the capital of France?")
        self.retriever.retrieve.assert_called_once()

if __name__ == "__main__":
    unittest.main()
