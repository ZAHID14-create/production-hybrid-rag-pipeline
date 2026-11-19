from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    """Evaluation question taxonomy representing distinct user query archetypes."""
    DIRECT_FACTUAL = "direct_factual"
    KEYWORD_BASED = "keyword_based"
    SEMANTIC = "semantic"
    MULTI_PASSAGE = "multi_passage"
    DIFFERENT_PAGES = "different_pages"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class RetrievalConfigName(str, Enum):
    """The six evaluated retrieval and reranking configurations."""
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RRF_BGE = "hybrid_rrf_bge"
    DENSE_BGE = "dense_bge"
    BM25_BGE = "bm25_bge"


class EvaluationQuestion(BaseModel):
    """A ground-truth annotated question within the evaluation benchmark dataset."""
    question_id: str = Field(..., description="Unique alphanumeric identifier (e.g. 'eval_q001')")
    question: str = Field(..., description="The query string submitted to the retrieval pipeline")
    expected_answer: str = Field(..., description="The verified ground-truth answer grounded in corpus facts")
    relevant_document_ids: List[str] = Field(default_factory=list, description="IDs of documents containing answers")
    relevant_chunk_ids: List[str] = Field(default_factory=list, description="IDs of chunks containing supporting evidence")
    relevant_page_numbers: List[int] = Field(default_factory=list, description="1-based page numbers containing supporting evidence")
    relevance_judgments: Dict[str, float] = Field(
        default_factory=dict,
        description="Graded relevance scores (e.g. chunk_id -> 1.0 for relevant, 2.0 for highly relevant)"
    )
    question_category: QuestionCategory = Field(..., description="Taxonomy category of the evaluation question")
    is_verified: bool = Field(default=False, description="Whether this question and ground-truth were human-verified")
    notes: Optional[str] = Field(default=None, description="Editorial or contextual notes regarding the question")


class EvaluationDataset(BaseModel):
    """Full container for the benchmark evaluation dataset."""
    dataset_version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    description: str = "Ground-truth benchmark dataset for Hybrid RAG retrieval, reranking, and generation evaluation."
    questions: List[EvaluationQuestion] = Field(default_factory=list)


class QueryRetrievalResult(BaseModel):
    """Detailed evaluation metrics for a single query under a specific configuration."""
    question_id: str
    question: str
    category: str
    is_verified: bool
    retrieved_chunk_ids: List[str]
    relevant_chunk_ids: List[str]
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    retrieval_time_ms: float
    reranking_time_ms: Optional[float] = None
    total_time_ms: float
    retrieved_chunks_preview: List[Dict[str, Any]] = Field(default_factory=list)


class ConfigurationSummary(BaseModel):
    """Aggregated macro metrics for one retrieval configuration across all evaluation queries."""
    configuration: RetrievalConfigName
    display_name: str
    description: str
    queries_evaluated: int
    queries_with_relevant: int
    queries_zero_relevant: int
    total_relevant_chunks: int
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    per_query_results: List[QueryRetrievalResult] = Field(default_factory=list)


class AnswerEvaluationResult(BaseModel):
    """Answer quality and citation faithfulness evaluation for a single query."""
    question_id: str
    question: str
    category: str
    expected_answer: str
    generated_answer: str
    has_sufficient_context: bool
    expected_sufficient_context: bool
    correct_insufficient_context_behavior: bool
    citations: List[int]
    invalid_citations: List[int]
    citation_validity: bool
    citation_precision: float
    citation_recall: float
    faithfulness_score: float
    answer_relevance_score: float
    unsupported_claims_detected: bool
    evaluation_notes: str


class PerformanceStageBenchmark(BaseModel):
    """Latency and resource breakdown for a specific stage in the pipeline."""
    stage_name: str
    sample_count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


class PerformanceBenchmarkReport(BaseModel):
    """Comprehensive performance benchmark covering all pipeline stages."""
    stages: List[PerformanceStageBenchmark]
    cold_start_latencies_ms: Dict[str, float]
    warm_latencies_ms: Dict[str, float]
    model_initialization_ms: Dict[str, float]
    peak_process_memory_mb: Optional[float] = None
    measurement_notes: str


class EvaluationRunRequest(BaseModel):
    """Request payload to trigger an evaluation benchmark run."""
    k_values: List[int] = Field(default=[1, 3, 5, 8], description="Cutoff depths for Precision@K, Recall@K, nDCG@K")
    include_answer_evaluation: bool = Field(default=True, description="Whether to evaluate answer quality & citations")
    use_cached_answers: bool = Field(default=True, description="Whether to reuse cached generation answers to save API usage")
    rerun_live_generation: bool = Field(default=False, description="Force live Gemini API generation")


class EvaluationReportResponse(BaseModel):
    """Complete evaluation report returned by the API and displayed in the frontend dashboard."""
    report_id: str
    timestamp: str
    corpus_document_count: int
    corpus_chunk_count: int
    corpus_document_names: List[str] = Field(default_factory=list, description="Names of documents in the evaluated corpus")
    dataset_total_questions: int
    dataset_verified_questions: int
    dataset_unverified_questions: int
    k_values: List[int]
    configurations: Dict[str, ConfigurationSummary]
    answer_evaluations: List[AnswerEvaluationResult] = Field(default_factory=list)
    performance: PerformanceBenchmarkReport
    best_retrieval_configuration: str
    best_configuration_rationale: str
    hardware_and_environment: Dict[str, Any]
    methodological_safeguards: List[str]
    evaluation_mode: str = Field(default="live", description="'benchmark' or 'live'")
    ground_truth_aligned: bool = Field(default=True, description="Whether corpus matches ground truth IDs")
    alignment_notice: Optional[str] = Field(default=None, description="Corpus-ground truth alignment explanation")
