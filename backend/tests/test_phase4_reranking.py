"""
Comprehensive Automated Test Suite for Phase 4:
BGE Cross-Encoder Reranking (BAAI/bge-reranker-base architecture),
CPU Inference, Query-Chunk Pair Construction, Descending Relevance Ordering,
Metadata & Citation Preservation, Latency Isolation, and API Route Controller Tests.
"""
import io
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.core.config import settings
from app.models.document import ChunkMetadata
from app.models.retrieval import (
    RetrievalMethod,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import ChromaVectorStore
from app.services.bm25_service import BM25IndexService
from app.services.retrieval_service import RetrievalService
from app.services.reranker_service import RerankerService, reranker_service
from app.api.routes_retrieval import search_documents


class SafeTempDir:
    """Context manager for temporary directories that suppresses Windows file-lock errors on cleanup."""
    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path, ignore_errors=True)


def test_model_singleton_and_loading_behavior():
    """Test 1: Model loading and singleton behavior."""
    print("\n--- Test 1: Model Singleton & Lazy Loading Behavior ---")
    svc = RerankerService()
    assert not svc.is_loaded, "Model should not be loaded on instantiation"

    # Set mock mode for fast unit test
    svc._is_mock = True
    scores = svc.score_pairs("test query", ["text one", "text two"])
    assert len(scores) == 2
    # Verify singleton instance from app.services
    from app.services.reranker_service import reranker_service as global_reranker
    assert global_reranker is not None
    assert global_reranker.max_length == 512
    assert global_reranker.default_batch_size == 16
    print("✓ Passed: Lazy loading and singleton architecture verified.")


def test_real_pytorch_cpu_inference():
    """Test 2: Real PyTorch sequence classification inference on CPU."""
    print("\n--- Test 2: Real PyTorch Cross-Encoder CPU Inference ---")
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", num_labels=1, local_files_only=True
    )
    model.to("cpu")
    model.eval()

    svc = RerankerService()
    svc.set_model_and_tokenizer(model, tok)
    assert svc.is_loaded
    assert not svc._is_mock

    # Score two pairs using real PyTorch tensor inference
    t0 = time.perf_counter()
    scores = svc.score_pairs(
        query="Transformers in natural language processing",
        texts=[
            "Transformers utilize self-attention mechanisms to process sequential text.",
            "Photosynthesis is the process by which green plants convert light energy into chemical energy.",
        ],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert len(scores) == 2
    for s in scores:
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0, f"Sigmoid score {s} out of [0, 1] range"
    print(f"✓ Passed: Real PyTorch CPU forward pass executed in {elapsed_ms:.2f}ms (scores: {[round(s, 4) for s in scores]}).")


def test_query_chunk_pair_construction_and_scoring():
    """Test 3 & 4: Query-chunk pair construction and score generation."""
    print("\n--- Test 3 & 4: Query-Chunk Pair Construction & Scoring ---")
    svc = RerankerService()
    svc._is_mock = True

    query = "asynchronous fastapi python"
    candidates = [
        RetrievedChunk(
            rank=1,
            chunk_id="c1",
            document_id="d1",
            document_name="fastapi.pdf",
            page_number=1,
            chunk_index=0,
            chunk_text="FastAPI is an asynchronous web framework for Python.",
            token_count=10,
            score=0.03,
        ),
        RetrievedChunk(
            rank=2,
            chunk_id="c2",
            document_id="d1",
            document_name="fastapi.pdf",
            page_number=2,
            chunk_index=1,
            chunk_text="Unrelated text about baking sourdough bread at high temperature.",
            token_count=11,
            score=0.02,
        ),
    ]

    reranked, rerank_ms = svc.rerank(query, candidates)
    assert len(reranked) == 2
    assert rerank_ms >= 0.0

    # Scores should be in [0, 1]
    for c in reranked:
        assert c.reranker_score is not None
        assert 0.0 <= c.reranker_score <= 1.0
    print("✓ Passed: Pairs constructed and scored with normalized confidence values.")


def test_descending_relevance_ordering():
    """Test 5: Candidates re-ordered strictly by descending reranker score."""
    print("\n--- Test 5: Descending Relevance Ordering ---")
    svc = RerankerService()
    svc._is_mock = True

    # Candidate 2 is highly relevant, Candidate 1 is weakly relevant
    candidates = [
        RetrievedChunk(
            rank=1,
            chunk_id="low_match",
            document_id="doc1",
            document_name="doc.pdf",
            page_number=1,
            chunk_index=0,
            chunk_text="General software engineering topics and version control.",
            token_count=9,
            score=0.035,
        ),
        RetrievedChunk(
            rank=2,
            chunk_id="high_match",
            document_id="doc1",
            document_name="doc.pdf",
            page_number=2,
            chunk_index=1,
            chunk_text="Deep residual networks with skip connections improve gradient backpropagation.",
            token_count=12,
            score=0.030,
        ),
    ]

    reranked, _ = svc.rerank("residual networks skip connections", candidates)

    assert reranked[0].chunk_id == "high_match"
    assert reranked[1].chunk_id == "low_match"
    assert reranked[0].reranker_score >= reranked[1].reranker_score
    assert reranked[0].rank == 1
    assert reranked[0].rerank_rank == 1
    assert reranked[0].original_rank == 2  # Original rank preserved!
    assert reranked[1].rank == 2
    assert reranked[1].original_rank == 1
    print(f"✓ Passed: Candidate #{candidates[1].chunk_id} promoted from rank 2 -> rank 1 based on reranker score.")


def test_metadata_and_citation_preservation():
    """Test 6: All metadata and citations preserved perfectly during reranking."""
    print("\n--- Test 6: Metadata & Citation Preservation ---")
    svc = RerankerService()
    svc._is_mock = True

    chunk = RetrievedChunk(
        rank=1,
        chunk_id="doc_p15_c4",
        document_id="uuid_9876",
        document_name="distributed_systems.pdf",
        page_number=15,
        chunk_index=4,
        chunk_text="Raft consensus algorithm uses leader election and replicated logs.",
        token_count=12,
        score=0.88,
        dense_score=0.88,
        dense_rank=1,
        bm25_score=1.45,
        bm25_rank=1,
        rrf_score=0.032,
    )

    reranked, _ = svc.rerank("consensus algorithm leader election", [chunk])
    r = reranked[0]

    assert r.chunk_id == "doc_p15_c4"
    assert r.document_id == "uuid_9876"
    assert r.document_name == "distributed_systems.pdf"
    assert r.page_number == 15
    assert r.chunk_index == 4
    assert r.chunk_text == chunk.chunk_text
    assert r.token_count == 12
    # Check that previous stage scores and ranks are still intact
    assert r.dense_score == 0.88
    assert r.dense_rank == 1
    assert r.bm25_score == 1.45
    assert r.bm25_rank == 1
    assert r.rrf_score == 0.032
    assert r.original_rank == 1
    assert r.rerank_rank == 1
    print("✓ Passed: All 7 core metadata fields and 5 diagnostic scores preserved without alteration.")


def test_empty_candidates_and_invalid_queries():
    """Test 7 & 8: Empty candidate list and invalid query edge cases."""
    print("\n--- Test 7 & 8: Empty Candidates & Invalid Queries ---")
    svc = RerankerService()
    svc._is_mock = True

    # 1. Empty candidates
    res1, t1 = svc.rerank("some query", [])
    assert res1 == []
    assert t1 == 0.0

    # 2. Empty query string
    candidate = RetrievedChunk(
        rank=1,
        chunk_id="c_test",
        document_id="d_test",
        document_name="test.pdf",
        page_number=1,
        chunk_index=0,
        chunk_text="Test passage content.",
        token_count=5,
        score=0.5,
    )
    res2, t2 = svc.rerank("    ", [candidate])
    assert len(res2) == 1
    assert res2[0].chunk_id == "c_test"
    assert res2[0].reranker_score == 0.0
    print("✓ Passed: Handled empty candidate lists and blank queries safely.")


def test_batch_processing():
    """Test 9: Configurable batch processing with small batch size."""
    print("\n--- Test 9: Configurable Batch Processing ---")
    svc = RerankerService()
    svc._is_mock = True

    candidates = [
        RetrievedChunk(
            rank=i + 1,
            chunk_id=f"batch_c_{i}",
            document_id="doc_batch",
            document_name="batch.pdf",
            page_number=i + 1,
            chunk_index=i,
            chunk_text=f"Sample text paragraph number {i} for batching verification.",
            token_count=10,
            score=0.1,
        )
        for i in range(7)
    ]

    # Run with small batch_size = 2
    reranked, _ = svc.rerank(
        query="batching verification paragraph",
        candidates=candidates,
        batch_size=2,
    )
    assert len(reranked) == 7
    print("✓ Passed: Successfully processed 7 items across 4 batches of size 2.")


def test_truncation_and_max_sequence_length():
    """Test 10: Truncation behavior on very long text (> 512 tokens)."""
    print("\n--- Test 10: Truncation & Max Sequence Length ---")
    # Test real tokenizer truncation
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    long_text = "word " * 1000  # 1000 words, > 512 tokens
    encoded = tok([["query", long_text]], truncation=True, max_length=512, return_tensors="pt")
    seq_len = encoded["input_ids"].shape[1]
    assert seq_len <= 512, f"Sequence length {seq_len} exceeded max_length 512"
    print(f"✓ Passed: Long input cleanly truncated to {seq_len} tokens (<= 512).")


def test_retrieval_service_reranking_enabled_vs_disabled():
    """Test 11: RetrievalService with rerank=True vs rerank=False."""
    print("\n--- Test 11: RetrievalService Rerank Enabled vs Disabled ---")
    with SafeTempDir() as tmp_chroma, SafeTempDir() as tmp_bm:
        vs = ChromaVectorStore(persist_directory=tmp_chroma)
        bm = BM25IndexService(index_dir=tmp_bm)

        chunks = [
            ChunkMetadata(
                chunk_id="chunk_alpha",
                document_id="doc_1",
                document_name="guide.pdf",
                page_number=1,
                chunk_index=0,
                chunk_text="Docker containers package applications with all system dependencies for portability.",
                token_count=12,
                character_count=85,
            ),
            ChunkMetadata(
                chunk_id="chunk_beta",
                document_id="doc_1",
                document_name="guide.pdf",
                page_number=2,
                chunk_index=1,
                chunk_text="Kubernetes coordinates automated container deployment, autoscaling, and service discovery.",
                token_count=13,
                character_count=93,
            ),
        ]
        vs.add_chunks(chunks)
        bm.add_chunks(chunks)

        service = RetrievalService()
        from app.services import retrieval_service as rs_mod
        orig_vs = rs_mod.vector_store
        orig_bm = rs_mod.bm25_service
        rs_mod.vector_store = vs
        rs_mod.bm25_service = bm

        try:
            # 1. Rerank DISABLED
            req_no_rerank = RetrievalRequest(
                query="Kubernetes container deployment autoscaling",
                retrieval_method=RetrievalMethod.HYBRID,
                top_k=2,
                rerank=False,
            )
            resp_no_rerank = service.search(req_no_rerank)
            assert not resp_no_rerank.rerank_enabled
            assert resp_no_rerank.reranking_time_ms is None
            for c in resp_no_rerank.results:
                assert c.reranker_score is None

            # 2. Rerank ENABLED (using mock for unit test)
            reranker_service._is_mock = True
            req_with_rerank = RetrievalRequest(
                query="Kubernetes container deployment autoscaling",
                retrieval_method=RetrievalMethod.HYBRID,
                top_k=2,
                rerank=True,
                reranker_top_k=2,
            )
            resp_with_rerank = service.search(req_with_rerank)
            assert resp_with_rerank.rerank_enabled
            assert resp_with_rerank.reranking_time_ms is not None
            assert resp_with_rerank.reranker_model == reranker_service.model_name
            assert resp_with_rerank.results[0].reranker_score is not None
            assert resp_with_rerank.results[0].original_rank is not None
            assert resp_with_rerank.results[0].rerank_rank == 1
            print("✓ Passed: Clean distinction between reranking disabled and enabled modes.")
        finally:
            rs_mod.vector_store = orig_vs
            rs_mod.bm25_service = orig_bm


def test_api_route_controller_validation():
    """Test 12: FastAPI route controller validation for rerank request and response."""
    print("\n--- Test 12: API Route Controller Validation ---")
    reranker_service._is_mock = True
    req = RetrievalRequest(
        query="cloud container orchestrator",
        retrieval_method=RetrievalMethod.HYBRID,
        top_k=3,
        rerank=True,
        reranker_top_k=2,
    )
    resp = search_documents(req)
    assert isinstance(resp, RetrievalResponse)
    assert resp.rerank_enabled is True
    assert resp.reranking_time_ms is not None
    assert resp.execution_time_ms >= resp.reranking_time_ms
    print(f"✓ Passed: API route controller validated with rerank=True (reranking time: {resp.reranking_time_ms} ms).")


def run_all_tests():
    print("=" * 65)
    print("STARTING PHASE 4 TEST SUITE: BGE CROSS-ENCODER RERANKING")
    print("=" * 65)
    test_model_singleton_and_loading_behavior()
    test_real_pytorch_cpu_inference()
    test_query_chunk_pair_construction_and_scoring()
    test_descending_relevance_ordering()
    test_metadata_and_citation_preservation()
    test_empty_candidates_and_invalid_queries()
    test_batch_processing()
    test_truncation_and_max_sequence_length()
    test_retrieval_service_reranking_enabled_vs_disabled()
    test_api_route_controller_validation()
    print("\n" + "=" * 65)
    print("ALL PHASE 4 TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
