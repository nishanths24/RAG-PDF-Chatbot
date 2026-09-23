"""Sidebar layout for the RAG PDF chatbot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


def render_sidebar() -> dict[str, Any]:
    """Render the application sidebar."""

    # --------------------------------------------------------
    # Document Library
    # --------------------------------------------------------

    st.sidebar.header("Document Library")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help=(
            "Upload a PDF document to index it "
            "for question answering."
        ),
    )

    st.sidebar.divider()

    # --------------------------------------------------------
    # Retrieval Settings
    # --------------------------------------------------------

    st.sidebar.subheader("Retrieval Settings")

    top_k = st.sidebar.slider(
        "Number of sources",
        min_value=2,
        max_value=10,
        value=6,
        step=1,
        help=(
            "Number of document chunks retrieved "
            "for each question."
        ),
    )

    st.sidebar.caption(
        f"Using the top {top_k} most relevant chunks."
    )

    st.sidebar.divider()

    # --------------------------------------------------------
    # Local AI Stack
    # --------------------------------------------------------

    st.sidebar.caption("Local AI Stack")

    st.sidebar.caption(
        "Embeddings: all-MiniLM-L6-v2"
    )

    st.sidebar.caption(
        "Vector DB: FAISS"
    )

    st.sidebar.caption(
        "LLM: Qwen2.5 3B"
    )

    st.sidebar.caption(
        "Runtime: Ollama"
    )

    st.sidebar.caption(
        "No paid API required"
    )

    return {
        "uploaded_file": uploaded_file,
        "top_k": top_k,
    }


def save_uploaded_file(
    uploaded_file: Any,
    upload_directory: Path,
) -> Path:
    """Save an uploaded PDF to the uploads directory."""

    upload_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = (
        upload_directory
        / uploaded_file.name
    )

    file_path.write_bytes(
        uploaded_file.getvalue()
    )

    return file_path