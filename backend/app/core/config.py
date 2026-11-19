from pathlib import Path
from typing import List, Union, Optional
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine absolute project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Project Information
    PROJECT_NAME: str = "Production Hybrid RAG Pipeline with Source Citations"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS Configuration
    ALLOWED_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Google Gemini API & Resilience Configuration
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.8-flash"
    GEMINI_FALLBACK_MODEL: str = "gemini-3.1-flash-lite"
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_MAX_OUTPUT_TOKENS: int = 1024
    GEMINI_TIMEOUT_SECONDS: float = 60.0
    GEMINI_MAX_RETRIES: int = 2
    GEMINI_RETRY_BASE_DELAY: float = 1.0
    GEMINI_RETRY_MAX_DELAY: float = 4.0

    # Document Upload & Ingestion Settings
    MAX_UPLOAD_SIZE_MB: int = 1024  # 1 GB upload limit
    MAX_UPLOAD_SIZE_BYTES: int = 1024 * 1024 * 1024  # 1 GB in bytes
    DOCUMENTS_DIRECTORY: str = str(BASE_DIR / "documents")
    PROCESSED_DATA_DIRECTORY: str = str(BASE_DIR / "data" / "processed_documents")

    # Storage & Indices Paths (Configurable for Phase 2B+)
    CHROMA_PERSIST_DIRECTORY: str = str(BASE_DIR / "data" / "chroma_db")
    BM25_INDEX_DIRECTORY: str = str(BASE_DIR / "data" / "bm25_index")

    # Text Chunking Settings
    CHUNK_SIZE: int = 500       # Target tokens per chunk
    CHUNK_OVERLAP: int = 75     # Overlap tokens between consecutive chunks
    TOKENIZER_ENCODING: str = "cl100k_base"

    # Hybrid Retrieval & Reranker Optimization Defaults (Phase 3 & 4)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    RRF_K: int = 60
    RERANKER_MAX_LENGTH: int = 384
    RERANKER_INT8_QUANTIZATION: bool = True
    RERANKER_CANDIDATE_POOL: int = 10
    RERANKER_NUM_THREADS: Optional[int] = None

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("CHROMA_PERSIST_DIRECTORY", "BM25_INDEX_DIRECTORY", "PROCESSED_DATA_DIRECTORY", "DOCUMENTS_DIRECTORY", mode="before")
    @classmethod
    def resolve_project_paths(cls, v: str) -> str:
        if isinstance(v, str) and (v.startswith("../") or v.startswith("./")):
            rel_part = v.lstrip("./").lstrip(".\\")
            if rel_part.startswith("data") or rel_part.startswith("documents"):
                return str(BASE_DIR / rel_part)
        return str(v)

    @model_validator(mode="after")
    def sync_upload_size_bytes(self) -> "Settings":
        """Ensures MAX_UPLOAD_SIZE_BYTES is synchronized with MAX_UPLOAD_SIZE_MB."""
        if self.MAX_UPLOAD_SIZE_MB > 0:
            self.MAX_UPLOAD_SIZE_BYTES = self.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        return self

    model_config = SettingsConfigDict(
        env_file=("backend/.env", ".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
