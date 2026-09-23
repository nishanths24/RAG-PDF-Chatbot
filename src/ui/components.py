"""Reusable Streamlit UI components."""

from __future__ import annotations

import streamlit as st


def render_status(message: str, *, status: str = "info") -> None:
    """Render a status message."""
    if status == "success":
        st.success(message)
    elif status == "warning":
        st.warning(message)
    elif status == "error":
        st.error(message)
    else:
        st.info(message)


def render_source(
    file_name: str,
    page: object,
    score: float | None = None,
) -> None:
    """Render a retrieved document source."""
    score_text = ""

    if score is not None:
        score_text = f" · Similarity: {score:.3f}"

    st.caption(
        f"📄 {file_name} · Page {page}{score_text}"
    )