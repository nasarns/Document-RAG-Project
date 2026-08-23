import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings
from app.core.extractors import DocumentExtractor, DocumentExtractionError
from app.core.chunker import TextChunker
from app.core.embeddings import embedding_manager
from app.core.vector_db import vector_db
from app.core.rag_engine import rag_engine

router = APIRouter(prefix="/api")


# Request & Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="User's natural language question")
    top_k: Optional[int] = Field(default=4, description="Number of context chunks to retrieve")
    doc_id_filter: Optional[str] = Field(default=None, description="Optional document ID filter")


class CitationModel(BaseModel):
    label: str
    filename: Optional[str] = None
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    section: Optional[str] = None
    score: float
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    insufficient_evidence: bool
    citations: List[CitationModel]
    retrieved_chunks: List[Dict[str, Any]]
    provider: str
    model: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    indexed_at: str
    chunk_count: int
    page_summary: str


class UploadResponse(BaseModel):
    message: str
    doc_id: str
    filename: str
    file_type: str
    total_blocks_extracted: int
    total_chunks_created: int


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app_name": "Document RAG Question Answering System",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "qdrant_collection": settings.QDRANT_COLLECTION_NAME
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document (PDF, DOCX, TXT, Excel, CSV), extracts text & metadata,
    generates embeddings, and indexes in Qdrant.
    """
    filename = file.filename or "uploaded_file"
    ext = Path(filename).suffix.lower()

    allowed_exts = [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".csv"]
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {', '.join(allowed_exts)}"
        )

    doc_id = str(uuid.uuid4())
    saved_filename = f"{doc_id}_{filename}"
    saved_path = settings.UPLOADS_DIR / saved_filename

    # 1. Save uploaded file to disk
    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # 2. Extract text and metadata
    try:
        extractor = DocumentExtractor()
        blocks = extractor.extract(saved_path)
    except DocumentExtractionError as e:
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")

    # 3. Chunk text
    chunker = TextChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = chunker.chunk_document_blocks(blocks, doc_id=doc_id)

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks could be extracted from document.")

    # 4. Generate Embeddings
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedding_manager.embed_texts(chunk_texts)

    # 5. Store in Qdrant Vector DB
    vector_db.upsert_chunks(chunks, embeddings)

    return UploadResponse(
        message="Document uploaded and indexed successfully.",
        doc_id=doc_id,
        filename=filename,
        file_type=ext.replace(".", ""),
        total_blocks_extracted=len(blocks),
        total_chunks_created=len(chunks)
    )


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Asks a question grounded in the indexed documents.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = rag_engine.query(
        question=request.question,
        top_k=request.top_k,
        doc_id_filter=request.doc_id_filter
    )
    return result


@router.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    """
    Returns list of all documents indexed in the vector store.
    """
    docs = vector_db.list_documents()
    return docs


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """
    Deletes an indexed document and all its chunks from Qdrant.
    """
    success = vector_db.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found or could not be deleted.")
    return {"message": f"Document '{doc_id}' deleted successfully."}


@router.post("/clear")
def clear_all():
    """
    Clears all documents and vectors from the database.
    """
    vector_db.clear_collection()
    return {"message": "All indexed documents have been cleared."}
