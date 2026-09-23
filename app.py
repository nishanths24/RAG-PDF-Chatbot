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
from src.rag.ollama_service import OllamaService
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

def build_services(
    top_k: int | None = None,
) -> tuple[DocumentService, ChatService]:
    """Create and connect all RAG services."""

    settings = get_settings()

    # --------------------------------------------------------
    # Embedding service
    # --------------------------------------------------------

    embedding_service = EmbeddingService(
        model=settings.embeddings.model,
    )

    # --------------------------------------------------------
    # FAISS vector store
    # --------------------------------------------------------

    vector_store = FAISSVectorStore(
        dimension=embedding_service.get_embedding_dimension(),
        persist_directory=settings.vectorstore_dir,
    )

    # --------------------------------------------------------
    # Document service
    # --------------------------------------------------------

    document_service = DocumentService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )

    # --------------------------------------------------------
    # Load persistent FAISS index
    # --------------------------------------------------------

    try:
        vector_store.load()
    except Exception:
        # No existing index is allowed.
        pass

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    actual_top_k = (
        top_k
        if top_k is not None
        else settings.retrieval.top_k
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=actual_top_k,
    )

    # --------------------------------------------------------
    # Ollama
    # --------------------------------------------------------

    ollama_service = OllamaService(
        model=settings.llm.model,
    )

    # --------------------------------------------------------
    # QA Chain
    # --------------------------------------------------------

    qa_chain = QAChain(
        retriever=retriever,
        llm=ollama_service,
    )

    # --------------------------------------------------------
    # Chat Service
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Streamlit page configuration
    # --------------------------------------------------------

    st.set_page_config(
        page_title="RAG PDF Chatbot",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --------------------------------------------------------
    # Session state
    # --------------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "indexed_file_hash" not in st.session_state:
        st.session_state["indexed_file_hash"] = None

    if "indexed_file_name" not in st.session_state:
        st.session_state["indexed_file_name"] = None

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    sidebar = render_sidebar()

    uploaded_file = sidebar["uploaded_file"]

    top_k = sidebar["top_k"]

    # --------------------------------------------------------
    # Page title
    # --------------------------------------------------------

    st.title("RAG PDF Chatbot")

    st.caption(
        "Ask questions about your PDF using local embeddings, "
        "FAISS, Ollama, and Qwen2.5 3B."
    )

    # --------------------------------------------------------
    # Build services
    # --------------------------------------------------------

    try:

        document_service, chat_service = build_services(
            top_k=top_k,
        )

    except Exception as exc:

        st.error(
            f"Unable to initialize the RAG system: {exc}"
        )

        st.stop()

    # --------------------------------------------------------
    # Upload and index PDF
    # --------------------------------------------------------

    if uploaded_file is not None:

        current_hash = get_file_hash(
            uploaded_file
        )

        previous_hash = (
            st.session_state.get(
                "indexed_file_hash"
            )
        )

        # Only process a new/different PDF.
        if current_hash != previous_hash:

            try:

                # Clear previous vector store.
                document_service.clear_index()

                # Clear previous chat history.
                st.session_state["messages"] = []

                # Save PDF.
                file_path = save_uploaded_file(
                    uploaded_file,
                    settings.uploads_dir,
                )

                # Index PDF.
                with st.spinner(
                    "Processing PDF..."
                ):

                    result = document_service.index_pdf(
                        file_path
                    )

                # Save session information.
                st.session_state[
                    "indexed_file_hash"
                ] = current_hash

                st.session_state[
                    "indexed_file_name"
                ] = uploaded_file.name

                st.session_state[
                    "indexed_document_id"
                ] = result.document_id

                st.success(
                    f"Indexed {result.file_name}: "
                    f"{result.pages} pages, "
                    f"{result.chunks} chunks."
                )

            except Exception as exc:

                st.error(
                    f"Unable to index PDF: {exc}"
                )

    # --------------------------------------------------------
    # Vector store status
    # --------------------------------------------------------

    if document_service.is_index_empty():

        st.info(
            "Upload a PDF from the sidebar "
            "to start asking questions."
        )

    else:

        st.success(
            f"Vector store ready — "
            f"{document_service.get_indexed_document_count()} "
            f"chunks indexed."
        )

    # --------------------------------------------------------
    # Chat interface
    # --------------------------------------------------------

    render_chat(
        chat_service
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()