# RAG PDF Chatbot

A local, privacy-friendly **Retrieval-Augmented Generation (RAG) PDF Chatbot** built with Python and Streamlit.

The application allows users to upload PDF documents, extract and chunk their content, create local semantic embeddings, store them in FAISS, retrieve relevant passages, and generate grounded answers using a locally running **Qwen2.5 3B** model through **Ollama**.

No paid LLM API is required.

---

## Overview

RAG PDF Chatbot combines semantic search with a local language model to answer questions from uploaded PDF documents.

Instead of sending the entire document to an LLM, the application:

1. Extracts text from the uploaded PDF.
2. Splits the document into manageable chunks.
3. Generates local embeddings using `all-MiniLM-L6-v2`.
4. Stores the embeddings in a persistent FAISS vector index.
5. Retrieves relevant chunks for each question.
6. Uses Qwen2.5 3B through Ollama to generate an answer.
7. Displays the retrieved document pages as sources.

The system also uses query-aware retrieval for broad document-level questions such as summaries and overviews.

---

## Architecture

```text
                    PDF Upload
                        │
                        ▼
                PDF Text Extraction
                     PyPDF
                        │
                        ▼
                  Text Chunking
             Recursive Character Splitter
                        │
                        ▼
              Local Text Embeddings
          all-MiniLM-L6-v2 / 384 dimensions
                        │
                        ▼
                 FAISS Vector DB
                  Persistent Index
                        │
                        ▼
                Query Processing
                        │
             ┌──────────┴──────────┐
             │                     │
       Specific Query        Broad Query
             │                     │
       Semantic Search       Query-Aware Search
             │                     │
             └──────────┬──────────┘
                        ▼
                Relevant Context
                        │
                        ▼
                 RAG Prompting
                        │
                        ▼
                 Qwen2.5 3B
                    Ollama
                        │
                        ▼
                 Grounded Answer
                        │
                        ▼
               Sources + Page Numbers
```

---

## Key Features

* 📄 Upload and process PDF documents
* 🔎 Semantic document retrieval
* 🧠 Local sentence-transformer embeddings
* 🗂️ Persistent FAISS vector database
* 🤖 Local Qwen2.5 3B language model
* 🖥️ Ollama local inference
* 💬 Streamlit chat interface
* 📚 Source and page attribution
* 🔄 Automatic re-indexing when a different PDF is uploaded
* 🎯 Query-aware retrieval for broad document questions
* 🛡️ No paid LLM API required
* 🔐 Document processing can run locally
* ⚙️ Configurable chunk size, overlap, and retrieval count
* 🧪 Automated unit test suite

---

## Technology Stack

| Component         | Technology               |
| ----------------- | ------------------------ |
| Language          | Python 3.12              |
| UI                | Streamlit                |
| PDF Processing    | PyPDF                    |
| Text Splitting    | LangChain Text Splitters |
| Embeddings        | Sentence Transformers    |
| Embedding Model   | `all-MiniLM-L6-v2`       |
| Vector Database   | FAISS                    |
| LLM               | Qwen2.5 3B               |
| Local LLM Runtime | Ollama                   |
| Configuration     | YAML + python-dotenv     |
| Testing           | Python `unittest`        |
| Version Control   | Git + GitHub             |

---

## RAG Pipeline

### 1. PDF Ingestion

Uploaded PDFs are processed page by page using PyPDF.

Each extracted page receives metadata such as:

* File name
* Page number
* Document ID
* Source information

### 2. Text Chunking

Extracted pages are split into smaller overlapping chunks using a recursive character text splitter.

Default configuration:

```yaml
chunk_size: 1000
chunk_overlap: 150
```

### 3. Local Embeddings

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model produces 384-dimensional embeddings locally.

### 4. FAISS Retrieval

Embeddings are stored in a persistent FAISS index using normalized vectors and inner-product similarity.

The vector store stores both:

* Document content
* Document metadata

### 5. Query-Aware Retrieval

The retriever supports two modes.

#### Standard retrieval

Used for specific questions such as:

```text
What technologies are used?
What is the main objective?
What is the system architecture?
```

The query is embedded and the most relevant chunks are retrieved from FAISS.

#### Broad-question retrieval

Used for questions such as:

```text
What is this document about?
Summarize this document.
What are the main topics?
Give me an overview.
```

Multiple retrieval perspectives are used to obtain a more representative set of document passages before generating the final answer.

### 6. Local LLM Generation

The retrieved context is passed to:

```text
Qwen2.5 3B
```

through:

```text
Ollama
```

The model is instructed to use the retrieved PDF content as the primary source of truth and avoid unsupported information.

---

## Project Structure

```text
RAG-PDF-Chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── config/
│   └── settings.yaml
│
├── data/
│   ├── uploads/
│   └── vectorstore/
│
├── src/
│   ├── config/
│   │   └── settings.py
│   │
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   ├── document_processor.py
│   │   ├── text_splitter.py
│   │   └── exceptions.py
│   │
│   ├── embeddings/
│   │   └── embedding_service.py
│   │
│   ├── vectorstore/
│   │   └── faiss_store.py
│   │
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   ├── rag/
│   │   ├── prompts.py
│   │   ├── qa_chain.py
│   │   └── ollama_service.py
│   │
│   ├── services/
│   │   ├── document_service.py
│   │   └── chat_service.py
│   │
│   ├── ui/
│   │   ├── sidebar.py
│   │   ├── chat.py
│   │   └── components.py
│   │
│   └── utils/
│
├── scripts/
│   ├── build_index.py
│   └── clear_index.py
│
├── tests/
│
└── .streamlit/
    └── config.toml
```

---

## Requirements

* Python 3.12
* Ollama
* Qwen2.5 3B
* Windows, Linux, or macOS
* Approximately 4 GB+ available RAM recommended for comfortable local operation

The project can run without an OpenAI API key.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nishanths24/RAG-PDF-Chatbot.git
cd RAG-PDF-Chatbot
```

### 2. Create a virtual environment

Windows:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama on your system and verify:

```powershell
ollama --version
```

### 5. Download Qwen2.5 3B

```powershell
ollama pull qwen2.5:3b
```

Verify the model:

```powershell
ollama list
```

### 6. Start Ollama

```powershell
ollama serve
```

If Ollama is already running, do not start a second server.

### CPU-only systems

If Ollama attempts to use an unsupported GPU configuration, the application can be run with CPU inference:

```powershell
$env:OLLAMA_LLM_LIBRARY="cpu"
```

Then start Ollama:

```powershell
ollama serve
```

---

## Running the Application

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Open:

```text
http://localhost:8502
```

### Usage

1. Upload a PDF from the sidebar.
2. Wait for indexing to complete.
3. Ask questions in the chat interface.
4. Open **View sources** to inspect retrieved pages.
5. Upload another PDF to replace the current document index.

Example questions:

```text
What is this document about?

What are the main objectives?

What technologies are used?

Summarize this document in 5 points.

Explain the methodology.

What are the key findings?

What are the conclusions?
```

---

## Configuration

Main configuration is stored in:

```text
config/settings.yaml
```

Example:

```yaml
application:
  name: RAG PDF Chatbot
  environment: development

llm:
  provider: ollama
  model: qwen2.5:3b
  temperature: 0.0

embeddings:
  provider: local
  model: sentence-transformers/all-MiniLM-L6-v2

retrieval:
  top_k: 4

chunking:
  chunk_size: 1000
  chunk_overlap: 150

paths:
  uploads: data/uploads
  vectorstore: data/vectorstore
```

The Streamlit sidebar allows the retrieval source count to be adjusted during use.

---

## Testing

The project includes unit tests covering:

* PDF loading
* PDF validation
* Document processing
* Text chunking
* Embedding integration
* FAISS persistence
* Similarity search
* Retrieval
* Prompt construction
* Ollama service behavior
* QA chain behavior
* Chat service
* Configuration-related functionality

Run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The current project test suite contains **70 tests**, all passing in the latest verified run.

---

## Privacy and Local Processing

The application is designed around local processing.

The current AI stack is:

```text
PDF
 ↓
Local Embeddings
 ↓
Local FAISS
 ↓
Local Ollama
 ↓
Local Qwen2.5 3B
```

No paid OpenAI API is required for the current implementation.

Uploaded documents and generated vector indexes are kept outside the Git repository through `.gitignore`.

---

## Error Handling

The application includes validation for conditions such as:

* Missing PDF files
* Empty PDFs
* Non-PDF files
* PDFs without extractable text
* Invalid chunk configuration
* Invalid retrieval configuration
* Vector dimension mismatches
* Corrupted or incomplete FAISS persistence
* Empty retrieval queries
* Empty RAG context
* Ollama connection failures

---

## Current Limitations

* Scanned/image-only PDFs require OCR support, which is not currently part of the pipeline.
* Local LLM response speed depends on available CPU/GPU resources.
* Very large documents may require additional retrieval and context-management strategies.
* The current application is designed primarily around one active uploaded PDF at a time.
* Broad-document summaries are generated from retrieved representative passages rather than feeding the entire document to the LLM.

---

## Future Improvements

Potential future enhancements include:

* OCR support for scanned PDFs
* Hybrid keyword + semantic retrieval
* Cross-encoder reranking
* Conversation memory
* Multi-document comparison
* Document history
* Streaming LLM responses
* Better citation formatting
* Table and image extraction
* Automated evaluation datasets
* Retrieval-quality metrics
* Docker deployment
* Cloud deployment with configurable LLM providers

---

## Why This Project?

This project demonstrates a practical implementation of a modern RAG system without depending on a paid hosted LLM API.

It combines:

* Document processing
* Natural language processing
* Semantic embeddings
* Vector databases
* Information retrieval
* Prompt engineering
* Local LLM inference
* Backend service architecture
* Streamlit application development
* Automated testing

The project is intended as a portfolio demonstration of building an end-to-end AI application rather than a simple chatbot wrapper.

---

## Author

**Nishanth S**

Computer Science & Engineering (AI & ML)

GitHub:

https://github.com/nishanths24

---

## License

Add an appropriate open-source license if you intend to distribute the project publicly.
