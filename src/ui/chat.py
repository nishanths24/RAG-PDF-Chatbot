"""Chat interface for the RAG PDF chatbot."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.services.chat_service import ChatService


def render_sources(
    sources: list[Any],
) -> None:
    """Display retrieved PDF sources."""

    if not sources:
        return

    with st.expander(
        "View sources",
        expanded=False,
    ):

        for result in sources:

            document = result.document

            metadata = (
                document.metadata
                or {}
            )

            file_name = metadata.get(
                "file_name",
                "Unknown document",
            )

            page = metadata.get(
                "page",
                "Unknown page",
            )

            score = float(
                result.score
            )

            st.caption(
                f"PDF {file_name} | "
                f"Page {page} | "
                f"Similarity: {score:.3f}"
            )


def ask_question(
    chat_service: ChatService,
    question: str,
) -> None:
    """Send a question to the RAG pipeline."""

    question = question.strip()

    if not question:
        return

    # Add user message.
    st.session_state["messages"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:

        with st.spinner(
            "Searching the document..."
        ):

            response = chat_service.ask(
                question
            )

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
            }
        )

    except Exception as exc:

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": (
                    "Unable to answer the question: "
                    f"{exc}"
                ),
                "sources": [],
            }
        )


def render_chat(
    chat_service: ChatService,
) -> None:
    """Render chat history and input."""

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # --------------------------------------------------------
    # Chat history
    # --------------------------------------------------------

    for message in st.session_state["messages"]:

        role = message["role"]

        content = message["content"]

        with st.chat_message(role):

            st.markdown(content)

            if role == "assistant":

                render_sources(
                    message.get(
                        "sources",
                        [],
                    )
                )

    # --------------------------------------------------------
    # Chat input
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about your uploaded PDF..."
    )

    if question:

        ask_question(
            chat_service,
            question,
        )

        st.rerun()