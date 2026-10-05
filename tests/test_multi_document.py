"""Tests for multi-document RAG capabilities."""

import unittest
from unittest.mock import Mock, patch
from dataclasses import asdict
from datetime import datetime, timezone

from src.services.document_registry import DocumentRegistry, DocumentMetadata
from src.services.document_service import DocumentService
from src.vectorstore.faiss_store import FAISSVectorStore
from langchain_core.documents import Document

class MultiDocumentTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        self.mock_embedding_service = Mock()
        self.mock_embedding_service.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        
        # Test registry
        self.registry = DocumentRegistry(self.temp_path)
        
        # Test FAISS
        self.faiss_store = FAISSVectorStore(dimension=3, persist_directory=self.temp_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_document_registry_operations(self):
        registry = DocumentRegistry(self.temp_path)
        
        doc = DocumentMetadata("id1", "file1.pdf", "hash1", 10, 50, "now")
        registry.add_document(doc)
        
        self.assertEqual(len(registry.get_all_documents()), 1)
        self.assertTrue(registry.contains_hash("hash1"))
        
        registry.remove_document("id1")
        self.assertEqual(len(registry.get_all_documents()), 0)
        self.assertFalse(registry.contains_hash("hash1"))

    def test_faiss_document_removal(self):
        doc1 = Document(page_content="Text 1", metadata={"document_id": "id1"})
        doc2 = Document(page_content="Text 2", metadata={"document_id": "id2"})
        
        self.faiss_store.add_documents([doc1, doc2], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.assertEqual(self.faiss_store.count(), 2)
        
        self.faiss_store.remove_document("id1")
        self.assertEqual(self.faiss_store.count(), 1)
        self.assertEqual(self.faiss_store._documents[0].metadata["document_id"], "id2")
        
        self.faiss_store.clear()
        self.assertEqual(self.faiss_store.count(), 0)

    @patch('src.services.document_service.process_pdf')
    @patch('src.services.document_service.split_documents')
    def test_document_service_prevents_duplicates(self, mock_split, mock_process):
        mock_process.return_value = [Document(page_content="Page 1", metadata={"document_id": "id1", "file_name": "test.pdf"})]
        mock_split.return_value = [Document(page_content="Chunk 1", metadata={"document_id": "id1"})]
        
        service = DocumentService(self.mock_embedding_service, self.faiss_store)
        
        # First index
        service.index_pdf("test.pdf", file_hash="hash1")
        
        # Second index should fail
        with self.assertRaises(ValueError) as context:
            service.index_pdf("test.pdf", file_hash="hash1")
        self.assertIn("already indexed", str(context.exception))

    def test_cross_document_retrieval(self):
        doc1 = Document(page_content="Machine learning in PDF 1", metadata={"document_id": "id1", "file_name": "f1.pdf"})
        doc2 = Document(page_content="Deep learning in PDF 2", metadata={"document_id": "id2", "file_name": "f2.pdf"})
        
        self.faiss_store.add_documents([doc1, doc2], [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]])
        
        results = self.faiss_store.similarity_search([1.0, 0.0, 0.0], k=2)
        self.assertEqual(len(results), 2)
        # Verify both documents are retrieved
        retrieved_ids = [res[0].metadata["document_id"] for res in results]
        self.assertIn("id1", retrieved_ids)
        self.assertIn("id2", retrieved_ids)

if __name__ == '__main__':
    unittest.main()
