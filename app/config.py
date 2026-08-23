import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base Directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM Settings
    LLM_PROVIDER: str = Field(default="groq", description="LLM provider: groq, openai, gemini, ollama")
    GROQ_API_KEY: str = Field(default="", description="Groq API key")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key")
    LLM_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Model name")
    LLM_BASE_URL: str = Field(default="", description="Optional custom base URL (e.g. for Ollama)")

    # Vector DB & Embeddings
    QDRANT_STORAGE_PATH: str = Field(default=str(BASE_DIR / "data" / "qdrant_storage"))
    QDRANT_COLLECTION_NAME: str = Field(default="document_rag")
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5")
    
    # RAG Parameters
    CHUNK_SIZE: int = Field(default=700)
    CHUNK_OVERLAP: int = Field(default=150)
    TOP_K_RESULTS: int = Field(default=4)
    SIMILARITY_THRESHOLD: float = Field(default=0.35)

    # Server Settings
    BACKEND_HOST: str = Field(default="127.0.0.1")
    BACKEND_PORT: int = Field(default=8000)

    # File Storage Paths
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"

settings = Settings()

# Ensure necessary directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(settings.QDRANT_STORAGE_PATH, exist_ok=True)
