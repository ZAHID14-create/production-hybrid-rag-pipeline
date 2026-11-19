from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class IndexStatusResponse(BaseModel):
    """Schema representing current operational status of the indexing orchestrator."""

    status: str = Field(description="Operational status ('idle', 'indexing', 'rebuilding', 'error')")
    last_action: str = Field(description="Description of the last performed indexing action")
    last_indexed_at: Optional[datetime] = Field(default=None, description="Timestamp of the most recent indexing operation")
    last_error: Optional[str] = Field(default=None, description="Last error message encountered if any")
    total_documents: int = Field(default=0, ge=0, description="Total ingested documents in catalog")
    total_chunks: int = Field(default=0, ge=0, description="Total chunks across all ingested documents")


class IndexStatsResponse(BaseModel):
    """Schema representing detailed statistics for both Dense Vector (ChromaDB) and BM25 indices."""

    # ChromaDB Vector Store
    chroma_status: str = Field(description="Status of ChromaDB collection ('ready', 'empty', 'error')")
    chroma_collection_name: str = Field(description="ChromaDB persistent collection name")
    chroma_chunk_count: int = Field(ge=0, description="Number of vector embeddings stored in ChromaDB")
    embedding_model_name: str = Field(description="HuggingFace model used for dense embeddings")
    embedding_dimension: int = Field(ge=1, description="Embedding vector dimensionality (e.g. 384)")

    # BM25 Lexical Index
    bm25_status: str = Field(description="Status of BM25Okapi lexical index ('ready', 'empty', 'error')")
    bm25_chunk_count: int = Field(ge=0, description="Number of text chunks indexed in BM25")
    bm25_corpus_size_bytes: int = Field(ge=0, description="Persistent corpus file size in bytes")

    # Unified Synchronization Metrics
    total_indexed_chunks: int = Field(ge=0, description="Total chunks currently indexed across pipelines")
    is_synchronized: bool = Field(description="Whether ChromaDB and BM25 chunk counts match perfectly")


class IndexRebuildResponse(BaseModel):
    """Schema returned after triggering a full index rebuild."""

    message: str = Field(description="Rebuild execution summary message")
    documents_processed: int = Field(ge=0, description="Number of documents read from storage")
    chunks_indexed: int = Field(ge=0, description="Total chunks embedded and added to both indices")
    duration_seconds: float = Field(ge=0.0, description="Elapsed rebuild time in seconds")
    status: str = Field(default="completed", description="Execution outcome status")
