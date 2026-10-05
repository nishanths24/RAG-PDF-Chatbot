"""Streamlit entry point for the RAG PDF Chatbot."""

from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.config.settings import get_settings
from src.embeddings.embedding_service import EmbeddingService
from src.rag.groq_service import GroqService
from src.rag.qa_chain import QAChain
from src.retrieval.retriever import Retriever
from src.services.chat_service import ChatService
from src.services.document_service import DocumentService
from src.ui.chat import render_chat
from src.ui.sidebar import (
    render_sidebar,
    save_uploaded_file,
)
from src.vectorstore.faiss_store import FAISSVectorStore


# ============================================================
# BUILD SERVICES
# ============================================================

@st.cache_resource
def build_services(
    top_k: int = 6,
) -> tuple[DocumentService, ChatService]:
    """Create and connect all RAG services."""

    settings = get_settings()

    embedding_service = EmbeddingService(
        model=settings.embeddings.model,
    )

    vector_store = FAISSVectorStore(
        dimension=embedding_service.get_embedding_dimension(),
        persist_directory=settings.vectorstore_dir,
    )

    document_service = DocumentService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )

    try:
        document_service.load_existing_index()
    except Exception:
        pass

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=top_k,
    )

    groq_service = GroqService(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
    )

    qa_chain = QAChain(
        retriever=retriever,
        llm=groq_service,
    )

    chat_service = ChatService(
        qa_chain
    )

    return document_service, chat_service


# ============================================================
# FILE HASH
# ============================================================

def get_file_hash(uploaded_file) -> str:
    """Generate a SHA-256 hash for the uploaded PDF."""
    return hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the Streamlit application."""

    settings = get_settings()

    st.set_page_config(
        page_title="RAG PDF Chatbot",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    sidebar = render_sidebar()

    uploaded_files = sidebar["uploaded_files"]
    top_k = sidebar["top_k"]

    st.title("RAG PDF Chatbot")

    st.caption(
        "Ask questions about your PDF using local embeddings, "
        "FAISS, and Groq."
    )

    try:
        document_service, chat_service = build_services(
            top_k=top_k,
        )
        chat_service.qa_chain.retriever.top_k = top_k
    except Exception as exc:
        st.error(f"Unable to initialize the RAG system: {exc}")
        st.stop()

    # Upload multiple PDFs
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_hash = get_file_hash(uploaded_file)
            
            if not document_service.registry.contains_hash(file_hash):
                try:
                    file_path = save_uploaded_file(
                        uploaded_file,
                        settings.uploads_dir,
                    )
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = document_service.index_pdf(file_path, file_hash=file_hash)
                    st.success(f"Indexed {result.file_name}: {result.pages} pages, {result.chunks} chunks.")
                except Exception as exc:
                    st.error(f"Unable to index PDF {uploaded_file.name}: {exc}")

    # UI for existing documents
    docs = document_service.get_all_documents()
    
    with st.sidebar:
        st.subheader("Indexed Documents")
        if docs:
            for doc in docs:
                st.markdown(f"☑ **{doc.file_name}**")
                st.caption(f"{doc.page_count} pages · {doc.chunk_count} chunks")
                if st.button("Remove", key=f"remove_{doc.document_id}"):
                    document_service.remove_document(doc.document_id)
                    st.rerun()
            
            st.divider()
            if st.button("Clear All Documents", type="primary"):
                document_service.clear_index()
                st.session_state["messages"] = []
                st.rerun()
        else:
            st.info("No documents uploaded yet.")

    # Status
    if document_service.is_index_empty():
        st.info("Upload PDFs from the sidebar to start asking questions.")
    else:
        st.success(
            f"Knowledge base ready — {len(docs)} documents · "
            f"{document_service.get_indexed_document_count()} chunks indexed."
        )

    # Chat
    render_chat(chat_service)


if __name__ == "__main__":
    main()