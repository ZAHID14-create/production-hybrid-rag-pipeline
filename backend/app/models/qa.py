from typing import List, Optional
from pydantic import BaseModel, Field
from app.models.retrieval import RetrievalMethod


class QARequest(BaseModel):
    """Request model for end-to-end question answering with citations."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question to answer using grounded document context.",
        examples=["How does multi-head attention work in transformers?"]
    )
    retrieval_method: RetrievalMethod = Field(
        default=RetrievalMethod.HYBRID,
        description="Retrieval strategy: 'dense' (ChromaDB), 'bm25' (Lexical), or 'hybrid' (RRF)."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of candidate passages to retrieve."
    )
    rerank: bool = Field(
        default=True,
        description="Whether to apply BAAI/bge-reranker-base cross-encoder reranking."
    )
    rrf_k: Optional[int] = Field(
        default=60,
        ge=1,
        le=200,
        description="Reciprocal Rank Fusion constant (used when retrieval_method is 'hybrid')."
    )
    reranker_top_k: Optional[int] = Field(
        default=3,
        ge=1,
        le=15,
        description="Number of top candidates to retain after reranking to feed into Gemini context."
    )
    temperature: Optional[float] = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for Gemini answer generation."
    )


class CitationSource(BaseModel):
    """Metadata and content for a single source passage used for citations."""

    citation_id: int = Field(..., description="1-based integer identifier (e.g. 1 for [1]).")
    citation_label: str = Field(..., description="Formatted bracket citation label (e.g. '[1]').")
    document_id: str = Field(..., description="Unique ID of the source PDF document.")
    document_name: str = Field(..., description="Original filename of the source document.")
    page_number: int = Field(..., description="1-indexed physical page number.")
    chunk_id: str = Field(..., description="Deterministic chunk identifier.")
    chunk_index: int = Field(..., description="0-indexed position within the document.")
    chunk_text: str = Field(..., description="Full text content of the retrieved chunk.")
    token_count: int = Field(default=0, description="Estimated token length of the chunk.")
    retrieval_rank: int = Field(..., description="Rank from initial retrieval (or original rank before rerank).")
    rerank_rank: Optional[int] = Field(default=None, description="Final rank after cross-encoder reranking.")
    score: Optional[float] = Field(default=None, description="Primary retrieval score.")
    dense_score: Optional[float] = Field(default=None, description="ChromaDB cosine similarity score.")
    dense_rank: Optional[int] = Field(default=None, description="Dense channel rank.")
    bm25_score: Optional[float] = Field(default=None, description="BM25 lexical score.")
    bm25_rank: Optional[int] = Field(default=None, description="BM25 channel rank.")
    rrf_score: Optional[float] = Field(default=None, description="Reciprocal Rank Fusion score.")
    reranker_score: Optional[float] = Field(default=None, description="BGE cross-encoder sigmoid confidence score.")


class QAResponse(BaseModel):
    """Grounded question answering response with citations and execution diagnostics."""

    query: str = Field(..., description="The original user question.")
    answer: str = Field(..., description="Natural language answer grounded in the sources.")
    retrieval_method: str = Field(..., description="Retrieval method used.")
    rerank_enabled: bool = Field(..., description="Whether cross-encoder reranking was applied.")
    citations: List[int] = Field(
        default_factory=list,
        description="List of valid citation IDs referenced in the answer (e.g. [1, 2])."
    )
    invalid_citations: List[str] = Field(
        default_factory=list,
        description="Any citation markers detected in the output that did not map to a real source."
    )
    sources: List[CitationSource] = Field(
        default_factory=list,
        description="All context passages supplied to the model mapped to citation IDs."
    )
    has_sufficient_context: bool = Field(
        default=True,
        description="False if no context was found or model indicated insufficient information."
    )
    total_execution_time_ms: float = Field(..., description="Total elapsed time from query to response.")
    retrieval_time_ms: float = Field(..., description="Time taken by retrieval stage.")
    embedding_time_ms: Optional[float] = Field(default=None, description="Query embedding inference time.")
    reranking_time_ms: Optional[float] = Field(default=None, description="BGE cross-encoder reranking time.")
    generation_time_ms: float = Field(..., description="Gemini API generation latency in milliseconds.")
    model_name: str = Field(..., description="Google Gemini model identifier used for generation.")
    is_fallback_used: bool = Field(default=False, description="Whether the fallback model was invoked due to primary unavailability.")
    retry_count: int = Field(default=0, description="Total retry attempts made before successful generation.")
    fallback_reason: Optional[str] = Field(default=None, description="Detailed explanation if fallback model was used.")
