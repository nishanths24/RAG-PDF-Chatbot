"""Streamlit entry point for RAG PDF Chatbot (Phase 1)."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings


def main() -> None:
    """Render the Phase 1 initialization screen."""
    settings = get_settings()

    st.set_page_config(
        page_title=settings.application.name,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title(settings.application.name)
    st.info(
        "Phase 1 initialization complete. "
        "The project foundation and configuration are loaded. "
        "PDF upload and RAG question answering are not implemented yet."
    )
    st.caption(f"Environment: {settings.application.environment}")


main()