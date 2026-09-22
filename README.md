# RAG PDF Chatbot

## Overview

RAG PDF Chatbot is a Streamlit application for asking questions about uploaded PDF documents using retrieval-augmented generation (RAG).

The project is currently in **Phase 1**. This phase establishes the production project layout, environment files, and a typed configuration system. PDF ingestion, embeddings, vector storage, retrieval, and chat are not implemented yet.

## Architecture

The intended architecture is a modular RAG pipeline:

1. Load PDF documents from a project-relative uploads directory.
2. Split documents into overlapping chunks.
3. Create embeddings with an OpenAI embedding model.
4. Persist vectors in a local FAISS index.
5. Retrieve the top-k chunks for a user question.
6. Generate an answer with an OpenAI chat model.

Phase 1 implements configuration and package structure only. The pipeline modules exist as placeholders for later phases.

## Features

Implemented in Phase 1:

- Project directory structure for ingestion, embeddings, vector store, retrieval, RAG, services, UI, scripts, and tests
- Typed settings loaded from `config/settings.yaml` and `.env`
- Path resolution relative to the project root
- Minimal Streamlit entry point that loads settings and displays the application name

Not implemented yet:

- PDF upload and ingestion
- Text splitting, embeddings, and FAISS indexing
- Retrieval and question answering
- Chat UI beyond the Phase 1 initialization screen

## Tech Stack

- Python 3.12
- Streamlit
- LangChain, langchain-openai, langchain-text-splitters
- FAISS (CPU)
- PyPDF
- python-dotenv, PyYAML, Pydantic
- tiktoken

`langchain-community` is installed for later phases and is not used in Phase 1.

## Project Structure

```text
RAG PDF Chatbot/
├── app.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── README.md
├── config/
│   └── settings.yaml
├── data/
│   ├── uploads/
│   └── vectorstore/
├── src/
│   ├── config/
│   ├── ingestion/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── retrieval/
│   ├── rag/
│   ├── services/
│   ├── ui/
│   └── utils/
├── scripts/
│   ├── build_index.py
│   └── clear_index.py
├── tests/
└── .streamlit/
    └── config.toml
```

## Installation

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

A virtual environment with the required packages is already present in this workspace.

## Configuration

Copy `.env.example` to `.env` and set values as needed:

```env
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=
OPENAI_EMBEDDING_MODEL=
```

Application defaults live in `config/settings.yaml`:

- `application.name`, `application.environment`
- `llm.provider`, `llm.model`, `llm.temperature`
- `embeddings.provider`, `embeddings.model`
- `retrieval.top_k`
- `chunking.chunk_size`, `chunking.chunk_overlap`
- `paths.uploads`, `paths.vectorstore`

`OPENAI_CHAT_MODEL` and `OPENAI_EMBEDDING_MODEL` override the YAML model names when they are non-empty. Paths in YAML must be relative to the project root. Phase 1 does not require a real OpenAI API key to start the Streamlit app.

## Running the Application

From the project root, with the virtual environment activated:

```bash
streamlit run app.py
```

The Phase 1 screen shows the application name and an initialization message. It does not accept PDFs or answer questions.

## RAG Pipeline

The RAG pipeline is planned for later phases. Current modules under `src/ingestion`, `src/embeddings`, `src/vectorstore`, `src/retrieval`, `src/rag`, and `src/services` raise `NotImplementedError` if called.

## Testing

Phase 1 tests document the current placeholder contracts using the standard library `unittest` runner:

```bash
python -m unittest discover -s tests -v
```

These tests assert that pipeline entry points are not implemented yet.

## Troubleshooting

- **Import errors when starting Streamlit:** run `streamlit run app.py` from the project root so `src` is importable.
- **Missing `.env`:** copy `.env.example` to `.env`. Phase 1 still starts with placeholder values.
- **Invalid YAML:** `src/config/settings.py` validates required sections and relative paths. Absolute Windows paths in `config/settings.yaml` are rejected.
- **Empty data directories after clone:** `data/uploads/.gitkeep` and `data/vectorstore/.gitkeep` keep the folders in version control; uploaded files and indexes are ignored.

## Future Improvements

- Implement PDF loading, chunking, and document processing
- Add embedding generation and FAISS persistence
- Wire retrieval and a LangChain QA chain using current package APIs
- Build Streamlit upload, sidebar, and chat UI
- Add scripts to build and clear the local index
- Expand tests beyond Phase 1 placeholder contracts
