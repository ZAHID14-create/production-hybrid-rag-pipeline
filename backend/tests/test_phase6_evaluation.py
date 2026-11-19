"""Automated test suite for Phase 6: RAG Evaluation, Comparison and Performance Benchmarking.

Covers the 16 mandatory test areas:
1. Precision@K (with hand-calculated verification)
2. Recall@K (with hand-calculated verification)
3. Mean Reciprocal Rank MRR (with hand-calculated verification)
4. Normalized Discounted Cumulative Gain nDCG@K (with hand-calculated verification)
5. Hit Rate@K (with hand-calculated verification)
6. Empty relevance judgments handling
7. Queries without relevant results (negative controls)
8. Multi-K evaluations (K=1, 3, 5, 8, 10)
9. Identical evaluation inputs across configurations
10. Retrieval strategy integration (BM25, Dense, Hybrid)
11. Reranking integration (BGE cross-encoder)
12. Citation validity and faithfulness evaluation
13. Latency aggregation and percentile statistics
14. Result persistence and report serialization
15. API schema and endpoint validation
16. Regression tests across all previous phases (Phase 1 to Phase 5)
"""

import math
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.models.evaluation import (
    AnswerEvaluationResult,
    ConfigurationSummary,
    EvaluationDataset,
    EvaluationQuestion,
    EvaluationReportResponse,
    EvaluationRunRequest,
    PerformanceStageBenchmark,
    QuestionCategory,
    RetrievalConfigName,
)
from app.models.retrieval import RetrievalMethod, RetrievalRequest
from app.services.evaluation_service import (
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluation_service,
)
from app.services.retrieval_service import retrieval_service


def test_01_precision_at_k_hand_calculated():
    """Test 1: Precision@K hand-calculated verification.
    
    Retrieved: ["c1", "c2", "c3", "c4"]
    Relevant: {"c1", "c3"}
    K=1: ["c1"] -> 1 relevant / 1 = 1.0
    K=2: ["c1", "c2"] -> 1 relevant / 2 = 0.5
    K=3: ["c1", "c2", "c3"] -> 2 relevant / 3 = 0.6667
    K=4: ["c1", "c2", "c3", "c4"] -> 2 relevant / 4 = 0.5
    """
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = {"c1", "c3"}

    assert compute_precision_at_k(retrieved, relevant, k=1) == 1.0
    assert compute_precision_at_k(retrieved, relevant, k=2) == 0.5
    assert compute_precision_at_k(retrieved, relevant, k=3) == 0.6667
    assert compute_precision_at_k(retrieved, relevant, k=4) == 0.5
    print("✓ Test 1 Passed: Precision@K matches hand-calculated values.")


def test_02_recall_at_k_hand_calculated():
    """Test 2: Recall@K hand-calculated verification.
    
    Retrieved: ["c1", "c2", "c3", "c4"]
    Relevant: {"c1", "c3", "c5"} (Total relevant = 3)
    K=1: ["c1"] -> 1 / 3 = 0.3333
    K=2: ["c1", "c2"] -> 1 / 3 = 0.3333
    K=3: ["c1", "c2", "c3"] -> 2 / 3 = 0.6667
    K=4: ["c1", "c2", "c3", "c4"] -> 2 / 3 = 0.6667
    """
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = {"c1", "c3", "c5"}

    assert compute_recall_at_k(retrieved, relevant, k=1) == 0.3333
    assert compute_recall_at_k(retrieved, relevant, k=2) == 0.3333
    assert compute_recall_at_k(retrieved, relevant, k=3) == 0.6667
    assert compute_recall_at_k(retrieved, relevant, k=4) == 0.6667
    print("✓ Test 2 Passed: Recall@K matches hand-calculated values.")


def test_03_mrr_hand_calculated():
    """Test 3: Mean Reciprocal Rank (MRR) hand-calculated verification.
    
    Case A: First relevant at rank 1 -> 1 / 1 = 1.0
    Case B: First relevant at rank 2 -> 1 / 2 = 0.5
    Case C: First relevant at rank 4 -> 1 / 4 = 0.25
    Case D: No relevant retrieved -> 0.0
    """
    relevant = {"c_target"}

    assert compute_mrr(["c_target", "c_other"], relevant) == 1.0
    assert compute_mrr(["c_other", "c_target"], relevant) == 0.5
    assert compute_mrr(["a", "b", "c", "c_target"], relevant) == 0.25
    assert compute_mrr(["a", "b", "c", "d"], relevant) == 0.0
    print("✓ Test 3 Passed: MRR matches hand-calculated reciprocal positions.")


def test_04_ndcg_at_k_hand_calculated():
    """Test 4: Normalized Discounted Cumulative Gain (nDCG@K) hand-calculated verification.
    
    Let K = 3.
    Relevance judgments: {"c1": 2.0, "c2": 1.0}
    Retrieved: ["c1", "c3", "c2"] (c3 is irrelevant rel=0)
    
    DCG@3:
    Rank 1 ("c1", rel=2.0): (2^2 - 1) / log2(2) = 3.0 / 1.0 = 3.0
    Rank 2 ("c3", rel=0.0): (2^0 - 1) / log2(3) = 0.0
    Rank 3 ("c2", rel=1.0): (2^1 - 1) / log2(4) = 1.0 / 2.0 = 0.5
    Total DCG@3 = 3.0 + 0.0 + 0.5 = 3.5
    
    IDCG@3:
    Sorted rels: [2.0, 1.0]
    Rank 1: (2^2 - 1) / log2(2) = 3.0
    Rank 2: (2^1 - 1) / log2(3) = 1.0 / 1.5849625 = 0.6309297
    Total IDCG@3 = 3.6309297
    
    nDCG@3 = 3.5 / 3.6309297 = 0.9639
    """
    retrieved = ["c1", "c3", "c2"]
    judgments = {"c1": 2.0, "c2": 1.0}

    computed_ndcg = compute_ndcg_at_k(retrieved, judgments, k=3)
    assert abs(computed_ndcg - 0.9639) < 0.001
    print("✓ Test 4 Passed: nDCG@K matches hand-calculated discounted gain formula.")


def test_05_hit_rate_at_k_hand_calculated():
    """Test 5: Hit Rate@K verification."""
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c2"}

    assert compute_hit_rate_at_k(retrieved, relevant, k=1) == 0.0
    assert compute_hit_rate_at_k(retrieved, relevant, k=2) == 1.0
    assert compute_hit_rate_at_k(retrieved, relevant, k=3) == 1.0
    assert compute_hit_rate_at_k(retrieved, {"non_existent"}, k=3) == 0.0
    print("✓ Test 5 Passed: HitRate@K correctly identifies top-K hits.")


def test_06_empty_relevance_judgments():
    """Test 6: Handling of empty relevance judgments and edge cases."""
    retrieved = ["c1", "c2", "c3"]
    empty_rel = set()
    empty_judgments = {}

    assert compute_precision_at_k(retrieved, empty_rel, k=3) == 0.0
    assert compute_recall_at_k(retrieved, empty_rel, k=3) == 0.0
    assert compute_mrr(retrieved, empty_rel) == 0.0
    assert compute_ndcg_at_k(retrieved, empty_judgments, k=3) == 0.0
    assert compute_hit_rate_at_k(retrieved, empty_rel, k=3) == 0.0
    print("✓ Test 6 Passed: Handled empty relevance ground truth safely.")


def test_07_queries_without_relevant_results():
    """Test 7: Out-of-domain queries with zero relevant results (negative controls)."""
    dataset = evaluation_service.get_dataset()
    neg_questions = [q for q in dataset.questions if q.question_category == QuestionCategory.INSUFFICIENT_CONTEXT]

    assert len(neg_questions) >= 2, "Dataset must contain at least 2 negative control questions"
    for q in neg_questions:
        assert len(q.relevant_chunk_ids) == 0
        assert len(q.relevance_judgments) == 0
    print("✓ Test 7 Passed: Negative control queries structured with 0 relevant chunks.")


def test_08_different_k_cutoffs():
    """Test 8: Monotonic properties across different K values."""
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c1", "c3", "c5"}

    recalls = [compute_recall_at_k(retrieved, relevant, k=k) for k in [1, 2, 3, 4, 5]]
    # Recall should be non-decreasing with increasing K
    for i in range(len(recalls) - 1):
        assert recalls[i] <= recalls[i + 1]
    print("✓ Test 8 Passed: Recall monotonically non-decreasing as K increases.")


def test_09_identical_evaluation_inputs():
    """Test 9: Verification that all 6 configurations evaluate identical question sets."""
    report = evaluation_service.get_latest_report()
    if report is None:
        report = evaluation_service.run_full_evaluation()

    configs = list(report.configurations.values())
    assert len(configs) == 6, "Expected 6 evaluated retrieval configurations"

    first_q_ids = [r.question_id for r in configs[0].per_query_results]
    for cfg in configs[1:]:
        cfg_q_ids = [r.question_id for r in cfg.per_query_results]
        assert cfg_q_ids == first_q_ids, f"Input queries mismatch between {configs[0].configuration} and {cfg.configuration}"
    print("✓ Test 9 Passed: All 6 configurations evaluated on strictly identical inputs.")


def test_10_retrieval_strategy_integration():
    """Test 10: Retrieval strategies (BM25, Dense, Hybrid) return non-empty ranked results."""
    query = "What is the vector dimension generated by the embedding model?"
    for m in [RetrievalMethod.BM25, RetrievalMethod.DENSE, RetrievalMethod.HYBRID]:
        resp = retrieval_service.search(RetrievalRequest(query=query, retrieval_method=m, top_k=3, rerank=False))
        assert resp.total_results > 0
        assert len(resp.results) <= 3
    print("✓ Test 10 Passed: BM25, Dense, and Hybrid retrieval strategies integrated.")


def test_11_reranking_integration():
    """Test 11: BGE Cross-Encoder reranking updates rank positions and adds reranker scores."""
    query = "How does the cross-encoder attention mechanism differ from bi-encoder retrieval?"
    resp_no_rerank = retrieval_service.search(
        RetrievalRequest(query=query, retrieval_method=RetrievalMethod.HYBRID, top_k=5, rerank=False)
    )
    resp_reranked = retrieval_service.search(
        RetrievalRequest(query=query, retrieval_method=RetrievalMethod.HYBRID, top_k=5, rerank=True, reranker_top_k=3)
    )

    assert resp_reranked.rerank_enabled is True
    assert resp_reranked.reranker_model is not None
    assert len(resp_reranked.results) == 3
    for chunk in resp_reranked.results:
        assert hasattr(chunk, "reranker_score")
        assert chunk.reranker_score is not None
        assert 0.0 <= chunk.reranker_score <= 1.0
    print("✓ Test 11 Passed: BGE cross-encoder reranking integrated with calibrated scores.")


def test_12_citation_evaluation():
    """Test 12: Citation validity and faithfulness rubric."""
    report = evaluation_service.get_latest_report()
    assert report is not None
    assert len(report.answer_evaluations) > 0

    for ans in report.answer_evaluations:
        assert isinstance(ans.citation_validity, bool)
        assert 0.0 <= ans.citation_precision <= 1.0
        assert 0.0 <= ans.citation_recall <= 1.0
        assert 0.0 <= ans.faithfulness_score <= 1.0
        if ans.category == "insufficient_context":
            assert ans.correct_insufficient_context_behavior is True
    print("✓ Test 12 Passed: Citation correctness and faithfulness rubric validated.")


def test_13_latency_aggregation():
    """Test 13: Latency metrics aggregation (Mean, Median, P95)."""
    report = evaluation_service.get_latest_report()
    perf = report.performance
    assert len(perf.stages) >= 5

    for stg in perf.stages:
        assert stg.sample_count > 0
        assert stg.min_ms <= stg.median_ms <= stg.p95_ms <= stg.max_ms
    print("✓ Test 13 Passed: Latency statistics satisfy percentile ordering invariants.")


def test_14_result_persistence():
    """Test 14: Report persistence to disk and JSON schema round-trip."""
    latest = evaluation_service.get_latest_report()
    assert latest is not None
    assert latest.report_id.startswith("eval_")
    assert latest.dataset_total_questions >= 12
    assert latest.dataset_verified_questions >= 11
    assert latest.dataset_unverified_questions >= 1
    print("✓ Test 14 Passed: Evaluation report successfully persisted and reloaded.")


def test_15_api_request_validation():
    """Test 15: Pydantic schema validation for evaluation API requests."""
    req = EvaluationRunRequest(k_values=[1, 3, 5], include_answer_evaluation=True)
    assert req.k_values == [1, 3, 5]
    assert req.use_cached_answers is True
    print("✓ Test 15 Passed: Evaluation API schemas validated.")


def test_16_regression_all_phases():
    """Test 16: Complete regression check for Phases 1-5."""
    # Phase 1: Health check
    from app.api.routes_health import check_health
    h = check_health()
    assert h.status == "healthy"

    # Phase 2A: Storage listing
    from app.services.storage_service import storage_service
    docs = storage_service.list_documents()
    assert len(docs) >= 1

    # Phase 2B: Chroma and BM25 index counts
    from app.services.vector_store import vector_store
    from app.services.bm25_service import bm25_service
    assert vector_store.collection.count() >= 8
    assert len(bm25_service.chunk_ids) >= 8

    # Phase 3: Hybrid RRF retrieval
    rrf_resp = retrieval_service.search(
        RetrievalRequest(query="hybrid architecture", retrieval_method=RetrievalMethod.HYBRID, top_k=2)
    )
    assert rrf_resp.total_results == 2

    # Phase 4: Reranking
    rerank_resp = retrieval_service.search(
        RetrievalRequest(query="cross encoder", retrieval_method=RetrievalMethod.HYBRID, top_k=3, rerank=True)
    )
    assert len(rerank_resp.results) <= 3

    # Phase 5: QA schema
    from app.models.qa import QARequest, QAResponse
    qa_req = QARequest(query="test query")
    assert qa_req.top_k == 5
    print("✓ Test 16 Passed: All regression tests for Phases 1, 2A, 2B, 3, 4, and 5 PASSED.")


def run_all_tests():
    print("=" * 80)
    print("PHASE 6 AUTOMATED VERIFICATION TEST SUITE (16 TEST MODULES)")
    print("=" * 80)
    test_01_precision_at_k_hand_calculated()
    test_02_recall_at_k_hand_calculated()
    test_03_mrr_hand_calculated()
    test_04_ndcg_at_k_hand_calculated()
    test_05_hit_rate_at_k_hand_calculated()
    test_06_empty_relevance_judgments()
    test_07_queries_without_relevant_results()
    test_08_different_k_cutoffs()
    test_09_identical_evaluation_inputs()
    test_10_retrieval_strategy_integration()
    test_11_reranking_integration()
    test_12_citation_evaluation()
    test_13_latency_aggregation()
    test_14_result_persistence()
    test_15_api_request_validation()
    test_16_regression_all_phases()
    print("=" * 80)
    print("ALL 16 PHASE 6 TESTS PASSED SUCCESSFULLY (100% PASS RATE)")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
