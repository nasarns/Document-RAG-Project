import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.config import settings
from app.core.embeddings import embedding_manager


class VectorDBManager:
    """
    Manages vector storage and similarity searches in Qdrant (local file-based persistence).
    """

    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.client = QdrantClient(path=settings.QDRANT_STORAGE_PATH)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            vector_dim = embedding_manager.vector_dimension
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
            )

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """
        Inserts or updates vector points in Qdrant.
        """
        if not chunks or not embeddings:
            return 0

        points = []
        for chunk, vector in zip(chunks, embeddings):
            payload = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "text": chunk["text"],
                "filename": chunk["metadata"].get("filename"),
                "file_type": chunk["metadata"].get("file_type"),
                "page_number": chunk["metadata"].get("page_number"),
                "sheet_name": chunk["metadata"].get("sheet_name"),
                "section": chunk["metadata"].get("section"),
                "chunk_index": chunk["metadata"].get("chunk_index"),
                "citation_label": chunk["metadata"].get("citation_label"),
                "indexed_at": datetime.now().isoformat()
            }
            points.append(
                PointStruct(
                    id=chunk["chunk_id"],
                    vector=vector,
                    payload=payload
                )
            )

        # Batch upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        doc_id_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches Qdrant for top-K most similar chunks to query vector.
        """
        query_filter = None
        if doc_id_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id_filter)
                    )
                ]
            )

        # Compatibility with qdrant_client query_points / search
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            ).points
        except Exception:
            # Fallback to search method if query_points not available
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )

        matched_chunks = []
        for hit in results:
            payload = hit.payload or {}
            matched_chunks.append({
                "chunk_id": payload.get("chunk_id", str(hit.id)),
                "doc_id": payload.get("doc_id"),
                "text": payload.get("text", ""),
                "score": float(hit.score),
                "filename": payload.get("filename"),
                "page_number": payload.get("page_number"),
                "sheet_name": payload.get("sheet_name"),
                "section": payload.get("section"),
                "citation_label": payload.get("citation_label"),
                "indexed_at": payload.get("indexed_at")
            })

        return matched_chunks

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Returns an aggregated summary of all indexed documents in Qdrant.
        """
        docs_summary: Dict[str, Dict[str, Any]] = {}
        offset = None

        while True:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            points, next_offset = scroll_result
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                if not doc_id:
                    continue

                if doc_id not in docs_summary:
                    docs_summary[doc_id] = {
                        "doc_id": doc_id,
                        "filename": payload.get("filename", "Unknown"),
                        "file_type": payload.get("file_type", "unknown"),
                        "indexed_at": payload.get("indexed_at", ""),
                        "chunk_count": 0,
                        "pages": set()
                    }

                docs_summary[doc_id]["chunk_count"] += 1
                page = payload.get("page_number")
                if page is not None:
                    docs_summary[doc_id]["pages"].add(page)

            if next_offset is None:
                break
            offset = next_offset

        # Format output
        result = []
        for doc in docs_summary.values():
            pages = sorted(list(doc["pages"]))
            page_info = f"{len(pages)} pages" if pages else "N/A"
            result.append({
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "file_type": doc["file_type"],
                "indexed_at": doc["indexed_at"],
                "chunk_count": doc["chunk_count"],
                "page_summary": page_info
            })

        return result

    def delete_document(self, doc_id: str) -> bool:
        """
        Deletes all chunks belonging to a document from Qdrant.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="doc_id",
                                match=MatchValue(value=doc_id)
                            )
                        ]
                    )
                )
            )
            return True
        except Exception:
            return False

    def clear_collection(self):
        """Clears all vectors in the collection."""
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection_exists()


# Global singleton instance
vector_db = VectorDBManager()
