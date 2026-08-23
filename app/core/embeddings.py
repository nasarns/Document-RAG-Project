from typing import List
import numpy as np
from fastembed import TextEmbedding
from app.config import settings


class EmbeddingManager:
    """
    Manages vector embeddings using FastEmbed (ONNX-powered Sentence Transformers / BGE).
    Provides fast, local, and CPU-efficient text and query vectorization.
    """

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        model_name = settings.EMBEDDING_MODEL
        # Fallback to standard BGE small if custom model not found
        try:
            self._model = TextEmbedding(model_name=model_name)
        except Exception:
            self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of document text chunks.
        """
        if not texts:
            return []
        embeddings_generator = self._model.embed(texts)
        return [emb.tolist() for emb in embeddings_generator]

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string for semantic search.
        """
        query_text = query.strip()
        embeddings_generator = self._model.embed([query_text])
        for emb in embeddings_generator:
            return emb.tolist()
        return []

    @property
    def vector_dimension(self) -> int:
        """Returns the embedding dimension for vector database schema."""
        sample = self.embed_query("dimension check")
        return len(sample)


# Global singleton instance
embedding_manager = EmbeddingManager()
