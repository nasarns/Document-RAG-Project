import streamlit as st
import requests
import json
from typing import List, Dict, Any

# API Endpoint Config
API_BASE_URL = "http://127.0.0.1:8000/api"

# Streamlit Page Configuration
st.set_page_config(
    page_title="Document AI Q&A System (RAG)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .citation-badge {
        display: inline-block;
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #C7D2FE;
    }
    .insufficient-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        color: #92400E;
        font-weight: 500;
        margin-bottom: 12px;
    }
    .chunk-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_doc_filter" not in st.session_state:
    st.session_state.active_doc_filter = None


# Helper API Functions
def check_backend_health():
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return res.status_code == 200, res.json() if res.status_code == 200 else {}
    except Exception:
        return False, {}


def fetch_documents() -> List[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def upload_document(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        res = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=60)
        return res.status_code == 200, res.json()
    except Exception as e:
        return False, {"detail": str(e)}


def delete_document(doc_id: str):
    try:
        res = requests.delete(f"{API_BASE_URL}/documents/{doc_id}", timeout=5)
        return res.status_code == 200
    except Exception:
        return False


def query_rag(question: str, top_k: int = 4, doc_filter: str = None):
    try:
        payload = {
            "question": question,
            "top_k": top_k,
            "doc_id_filter": doc_filter if doc_filter != "All Documents" else None
        }
        res = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=60)
        if res.status_code == 200:
            return res.json()
        else:
            return {"error": res.json().get("detail", "Error processing request")}
    except Exception as e:
        return {"error": f"Failed to connect to backend: {str(e)}"}


# ==========================================
# Sidebar: Upload & Document Management
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/search-database.png", width=70)
    st.title("Document Manager")

    # Backend Connection Status
    is_healthy, health_info = check_backend_health()
    if is_healthy:
        st.success(f"🟢 Connected ({health_info.get('llm_provider', 'LLM').upper()})")
    else:
        st.error("🔴 Backend Disconnected. Please ensure FastAPI server is running on port 8000.")

    st.markdown("---")

    # File Upload Section
    st.subheader("📤 Upload Documents")
    uploaded_file = st.file_uploader(
        "Supported formats: PDF, DOCX, TXT, Excel (.xlsx), CSV",
        type=["pdf", "docx", "txt", "xlsx", "xls", "csv"],
        help="Upload multi-format documents to index into Qdrant vector database."
    )

    if uploaded_file is not None:
        if st.button("🚀 Index Document", use_container_width=True):
            with st.spinner(f"Extracting & Vectorizing '{uploaded_file.name}'..."):
                success, resp = upload_document(uploaded_file)
                if success:
                    st.success(f"✅ Indexed {resp.get('total_chunks_created', 0)} chunks!")
                    st.rerun()
                else:
                    st.error(f"❌ Upload Failed: {resp.get('detail', 'Unknown error')}")

    st.markdown("---")

    # Indexed Documents Section
    st.subheader("📑 Indexed Knowledge Base")
    docs = fetch_documents()

    if not docs:
        st.info("No documents indexed yet. Upload one above to begin!")
    else:
        st.caption(f"Total Documents: {len(docs)}")
        for doc in docs:
            col_info, col_del = st.columns([4, 1])
            with col_info:
                st.write(f"📄 **{doc['filename']}**")
                st.caption(f"Chunks: {doc['chunk_count']} | Pages: {doc['page_summary']}")
            with col_del:
                if st.button("🗑️", key=f"del_{doc['doc_id']}", help="Delete from vector store"):
                    if delete_document(doc["doc_id"]):
                        st.success("Deleted!")
                        st.rerun()

    st.markdown("---")

    # Query Settings
    st.subheader("⚙️ Search Settings")
    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=4)

    doc_options = ["All Documents"] + [d["filename"] for d in docs]
    selected_doc = st.selectbox("Search Scope", doc_options)
    doc_id_map = {d["filename"]: d["doc_id"] for d in docs}
    selected_doc_id = doc_id_map.get(selected_doc, None)


# ==========================================
# Main Chat & Question Answering Area
# ==========================================
st.markdown('<div class="main-header">Document-Based AI Question Answering</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Retrieval-Augmented Generation (RAG) powered by <b>FastAPI</b>, <b>PyMuPDF</b>, <b>Sentence Transformers</b>, and <b>Qdrant</b>.</div>',
    unsafe_allow_html=True
)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Citations if present
        if "citations" in msg and msg["citations"]:
            st.markdown("**📌 Sources & Citations:**")
            badges = "".join([
                f'<span class="citation-badge">📄 {c["label"]} (Score: {c.get("score", "")})</span>'
                for c in msg["citations"]
            ])
            st.markdown(badges, unsafe_allow_html=True)

        # Display Expandable Retrieved Chunks
        if "retrieved_chunks" in msg and msg["retrieved_chunks"]:
            with st.expander("🔍 View Retrieved Evidence Chunks"):
                for idx, chunk in enumerate(msg["retrieved_chunks"], 1):
                    score = round(chunk.get("score", 0.0), 3)
                    label = chunk.get("citation_label", f"Chunk {idx}")
                    st.markdown(f"**Source {idx}: `{label}`** (Cosine Similarity: `{score}`)")
                    st.info(chunk.get("text", ""))


# User Query Input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process RAG Query
    with st.chat_message("assistant"):
        with st.spinner("Searching document vectors and generating grounded response..."):
            response_data = query_rag(
                question=prompt,
                top_k=top_k,
                doc_filter=selected_doc_id
            )

        if "error" in response_data:
            st.error(response_data["error"])
        else:
            answer = response_data.get("answer", "")
            is_insufficient = response_data.get("insufficient_evidence", False)
            citations = response_data.get("citations", [])
            chunks = response_data.get("retrieved_chunks", [])

            # Format answer
            if is_insufficient:
                st.markdown(
                    f'<div class="insufficient-box">⚠️ <b>Insufficient Evidence</b><br>{answer}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(answer)

            # Render Citations
            if citations:
                st.markdown("**📌 Sources & Citations:**")
                badges = "".join([
                    f'<span class="citation-badge">📄 {c["label"]} (Score: {c.get("score", "")})</span>'
                    for c in citations
                ])
                st.markdown(badges, unsafe_allow_html=True)

            # Render Collapsible Chunks
            if chunks:
                with st.expander("🔍 View Retrieved Evidence Chunks"):
                    for idx, chunk in enumerate(chunks, 1):
                        score = round(chunk.get("score", 0.0), 3)
                        label = chunk.get("citation_label", f"Chunk {idx}")
                        st.markdown(f"**Source {idx}: `{label}`** (Cosine Similarity: `{score}`)")
                        st.info(chunk.get("text", ""))

            # Save assistant message to session
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "retrieved_chunks": chunks,
                "insufficient_evidence": is_insufficient
            })
