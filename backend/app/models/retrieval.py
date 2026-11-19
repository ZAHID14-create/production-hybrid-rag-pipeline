from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class RetrievalMethod(str, Enum):
    """Supported retrieval strategies."""
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class RetrievalRequest(BaseModel):
    """Schema for retrieval search request."""

    query: str = Field(..., min_length=1, description="Natural language search query")
    retrieval_method: RetrievalMethod = Field(
        default=RetrievalMethod.HYBRID,
        description="Retrieval strategy: 'bm25', 'dense', or 'hybrid'",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of candidate chunks to retrieve from the base search stage",
    )
    rrf_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Smoothing constant for Reciprocal Rank Fusion (defaults to 60)",
    )
    rerank: bool = Field(
        default=False,
        description="Whether to apply BGE cross-encoder reranking to retrieved candidates",
    )
    reranker_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Final number of top reranked chunks to return (defaults to top_k if omitted)",
    )
    reranker_batch_size: Optional[int] = Field(
        default=16,
        ge=1,
        le=64,
        description="Inference batch size for cross-encoder reranking on CPU",
    )


class RetrievedChunk(BaseModel):
    """Schema for a single retrieved and ranked document chunk with citation metadata."""

    rank: int = Field(ge=1, description="1-indexed ranking position in final result list")
    chunk_id: str = Field(description="Deterministic unique chunk identifier")
    document_id: str = Field(description="Parent document unique identifier (UUID)")
    document_name: str = Field(description="Sanitized original document filename")
    page_number: int = Field(ge=1, description="1-indexed source PDF page number")
    chunk_index: int = Field(ge=0, description="0-indexed chunk position within parent document")
    chunk_text: str = Field(description="Extracted text content of this chunk")
    token_count: int = Field(ge=0, description="Estimated subword token count")
    score: float = Field(
        description="Primary score for this retrieval method (RRF score for hybrid, cosine similarity for dense, BM25 score for bm25, or reranker_score if reranked)"
    )
    dense_score: Optional[float] = Field(
        default=None,
        description="Cosine similarity score from dense vector search (range [0.0, 1.0])",
    )
    dense_rank: Optional[int] = Field(
        default=None,
        description="1-indexed rank in dense retrieval candidate list, or null if not in top dense candidates",
    )
    bm25_score: Optional[float] = Field(
        default=None,
        description="BM25Okapi relevance score from lexical search",
    )
    bm25_rank: Optional[int] = Field(
        default=None,
        description="1-indexed rank in BM25 candidate list, or null if not in top BM25 candidates",
    )
    rrf_score: Optional[float] = Field(
        default=None,
        description="Combined Reciprocal Rank Fusion score: sum(1 / (k + rank))",
    )
    original_rank: Optional[int] = Field(
        default=None,
        description="Initial 1-indexed retrieval rank position before cross-encoder reranking",
    )
    rerank_rank: Optional[int] = Field(
        default=None,
        description="1-indexed rank position after cross-encoder reranking",
    )
    reranker_score: Optional[float] = Field(
        default=None,
        description="Relevance score output by cross-encoder model (e.g. BAAI/bge-reranker-base)",
    )


class RetrievalResponse(BaseModel):
    """Comprehensive response payload for a retrieval query."""

    query: str = Field(description="The original search query executed")
    retrieval_method: str = Field(description="Retrieval method used: 'bm25', 'dense', or 'hybrid'")
    total_results: int = Field(ge=0, description="Count of chunks returned")
    results: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Ranked list of retrieved chunks with metadata and citations",
    )
    execution_time_ms: float = Field(
        ge=0.0,
        description="Total round-trip retrieval execution time in milliseconds",
    )
    embedding_time_ms: Optional[float] = Field(
        default=None,
        description="Query dense vector embedding computation time in milliseconds (for dense and hybrid)",
    )
    dense_candidates_count: Optional[int] = Field(
        default=None,
        description="Count of candidate chunks fetched from dense ChromaDB vector search",
    )
    bm25_candidates_count: Optional[int] = Field(
        default=None,
        description="Count of candidate chunks fetched from BM25 lexical search",
    )
    rrf_k: Optional[int] = Field(
        default=None,
        description="Reciprocal Rank Fusion constant k applied during hybrid fusion",
    )
    rerank_enabled: bool = Field(
        default=False,
        description="Whether BGE cross-encoder reranking was applied to the candidates",
    )
    reranking_time_ms: Optional[float] = Field(
        default=None,
        description="Execution time spent on cross-encoder reranking in milliseconds",
    )
    reranker_model: Optional[str] = Field(
        default=None,
        description="Model name of the cross-encoder used (e.g. 'BAAI/bge-reranker-base')",
    )
