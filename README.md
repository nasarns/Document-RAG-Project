# Document-Based AI Question Answering System Using RAG

An enterprise-grade, document-grounded AI Question Answering system built with **FastAPI**, **Qdrant Vector Database**, **Sentence Transformers / FastEmbed**, **PyMuPDF**, and **Streamlit**.

---

## 🎯 Project Overview & Objectives
- **Multi-Format Extraction**: Ingest **PDF**, **DOCX**, **TXT**, **Excel (.xlsx)**, and **CSV** files.
- **Citation-Aware Chunking**: Preserves exact document page numbers, sections, or table rows with every chunk.
- **Semantic Vector Storage**: Embeds chunks using FastEmbed/Sentence-Transformers and stores them in local persistent **Qdrant**.
- **Evidence-Grounded Generation**: Restricts the LLM to only answer based on retrieved document chunks.
- **Honest "Insufficient Evidence" Guardrail**: Returns an explicit *"Insufficient Evidence"* warning when the document doesn't contain the requested information, avoiding hallucinations.
- **Interactive UI**: Clean Streamlit dashboard with drag-and-drop uploads, chat history, citation badges, and expandable source views.

---

## 🛠️ Technology Stack
| Layer | Technology |
|---|---|
| **Language** | Python 3.12+ / 3.14 |
| **Backend API** | FastAPI, Uvicorn |
| **PDF Extraction** | PyMuPDF (`fitz`) |
| **DOCX Extraction** | `python-docx` |
| **Spreadsheet Extraction** | `pandas`, `openpyxl` |
| **Embeddings** | `fastembed` / `sentence-transformers` (`BAAI/bge-small-en-v1.5`) |
| **Vector Database** | `qdrant-client` (Local disk mode in `./data/qdrant_storage`) |
| **LLM Inference** | Groq (Llama 3.3 / Llama 3), OpenAI, Gemini, or Local Ollama |
| **Frontend UI** | Streamlit |
| **IDE & Tooling** | VS Code, Git |

---

## 📁 Project Directory Structure
```text
Document-RAG-Project/
├── .vscode/
│   ├── launch.json               # 1-click VS Code run/debug configurations
│   └── settings.json             # Python interpreter configuration
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic Settings & environment variables
│   ├── core/
│   │   ├── extractors.py         # Multi-format document extractors (PDF, DOCX, TXT, Excel, CSV)
│   │   ├── chunker.py            # Text chunker with citation metadata tracking
│   │   ├── embeddings.py         # FastEmbed vectorizer singleton
│   │   ├── vector_db.py          # Qdrant client & vector operations
│   │   └── rag_engine.py         # Grounded prompt builder, LLM caller & citations
│   ├── api/
│   │   └── routes.py             # FastAPI REST endpoints (/upload, /query, /documents, /health)
│   └── main.py                   # FastAPI application entry point
├── frontend/
│   └── app.py                    # Streamlit interactive UI
├── data/
│   ├── uploads/                  # Uploaded raw files
│   └── qdrant_storage/           # Local Qdrant database persistence
├── run.py                        # Single unified launcher script
├── requirements.txt              # Project dependencies
├── .env.example                  # Environment template
└── README.md                     # Documentation
```

---

## ⚡ Quick Start Guide (VS Code)

### 1. Configure API Key
Open `.env` in the project root and set your preferred LLM provider:

#### Option A: Groq (Recommended - Free & Ultra Fast)
1. Get a free key at [console.groq.com](https://console.groq.com/keys)
2. In `.env`:
   ```env
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_your_groq_api_key_here
   LLM_MODEL=llama-3.3-70b-versatile
   ```

#### Option B: OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your_openai_key_here
LLM_MODEL=gpt-4o-mini
```

#### Option C: Local Ollama (No API Key Required)
```env
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
```

---

### 2. Run the Application
In your VS Code terminal, run:
```bash
.\.venv\Scripts\python.exe run.py
```

Or simply press **F5** in VS Code to launch with the pre-configured debugger!

- 🌐 **Frontend UI**: [http://localhost:8501](http://localhost:8501)
- 📑 **Backend Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Testing Multi-Format Ingestion
1. Open [http://localhost:8501](http://localhost:8501).
2. In the sidebar, drag and drop any **PDF**, **DOCX**, **TXT**, **Excel (.xlsx)**, or **CSV** file.
3. Click **"🚀 Index Document"**.
4. Type your question in the chat input (e.g., *"What is the main finding in section 2?"*).
5. Review the grounded answer, the source citation tag (e.g. `📄 sample.pdf (Page 3)`), and click **"🔍 View Retrieved Evidence Chunks"** to view the exact text snippets retrieved from Qdrant.
6. Ask an out-of-scope question (e.g., *"What is the weather on Mars?"*) to verify the **"Insufficient Evidence"** safeguard.
