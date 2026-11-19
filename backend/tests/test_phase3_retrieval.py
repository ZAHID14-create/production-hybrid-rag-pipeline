"""
Comprehensive Automated Test Suite for Phase 3:
Hybrid Retrieval Using Reciprocal Rank Fusion (RRF),
Dense Semantic Retrieval (ChromaDB), BM25 Lexical Retrieval,
Mathematical Verification, and API Route Controller Tests.
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
from app.services.index_manager import IndexManager
from app.services.retrieval_service import RetrievalService, compute_rrf_score
from app.api.routes_retrieval import search_documents
import pymupdf


class SafeTempDir:
    """Context manager for temporary directories that suppresses Windows file-lock errors on cleanup."""
    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path, ignore_errors=True)


def test_rrf_mathematical_formula_verification():
    """Test 1: Pure mathematical verification of RRF formula against manual calculations."""
    print("\n--- Test 1: RRF Mathematical Formula Verification ---")
    k = 60

    # Case A: Item ranked #1 in Dense and #2 in BM25
    # Manual: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    expected_a = (1.0 / 61.0) + (1.0 / 62.0)  # ~0.032522474928
    actual_a = compute_rrf_score([1, 2], k=k)
    assert abs(actual_a - expected_a) < 1e-9, f"Case A failed: {actual_a} != {expected_a}"

    # Case B: Item ranked #1 in Dense only
    # Manual: 1/(60+1) = 1/61
    expected_b = 1.0 / 61.0  # ~0.016393442622
    actual_b = compute_rrf_score([1], k=k)
    assert abs(actual_b - expected_b) < 1e-9, f"Case B failed: {actual_b} != {expected_b}"

    # Case C: Item ranked #10 in BM25 only
    # Manual: 1/(60+10) = 1/70
    expected_c = 1.0 / 70.0  # ~0.014285714285
    actual_c = compute_rrf_score([10], k=k)
    assert abs(actual_c - expected_c) < 1e-9, f"Case C failed: {actual_c} != {expected_c}"

    # Case D: Verification of custom k parameter (e.g. k=20)
    expected_d = (1.0 / 21.0) + (1.0 / 25.0)
    actual_d = compute_rrf_score([1, 5], k=20)
    assert abs(actual_d - expected_d) < 1e-9, f"Case D failed: {actual_d} != {expected_d}"

    # Edge cases: empty ranks or non-positive
    assert compute_rrf_score([], k=k) == 0.0
    assert compute_rrf_score([0, -1], k=k) == 0.0
    print(f"✓ Passed: Exact match on all manual RRF calculations (e.g. [1, 2] -> {actual_a:.10f}).")


def test_dense_retrieval_standalone():
    """Test 2: Dense semantic retrieval against ChromaDB."""
    print("\n--- Test 2: Dense Semantic Retrieval Standalone ---")
    with SafeTempDir() as tmp_chroma:
        vs = ChromaVectorStore(persist_directory=tmp_chroma)
        chunks = [
            ChunkMetadata(
                chunk_id="chunk_attention",
                document_id="doc_ai",
                document_name="ai_handbook.pdf",
                page_number=1,
                chunk_index=0,
                chunk_text="Transformers rely on multi-head attention to capture long-range contextual relationships.",
                token_count=13,
                character_count=87,
            ),
            ChunkMetadata(
                chunk_id="chunk_cnn",
                document_id="doc_ai",
                document_name="ai_handbook.pdf",
                page_number=2,
                chunk_index=1,
                chunk_text="Convolutional kernels perform spatial convolutions across pixel channels for image classification.",
                token_count=13,
                character_count=98,
            ),
        ]
        vs.add_chunks(chunks)

        service = RetrievalService()
        # Temporarily use our isolated vector store
        old_vs = service.__dict__.get("vector_store")
        from app.services import retrieval_service as rs_mod
        orig_vs = rs_mod.vector_store
        rs_mod.vector_store = vs

        try:
            results, emb_time, count = service.retrieve_dense(
                "How does self-attention mechanism work in transformers?", top_k=2
            )
            assert len(results) == 2
            assert count == 2
            assert results[0].chunk_id == "chunk_attention"
            assert results[0].dense_rank == 1
            assert results[0].dense_score is not None and results[0].dense_score > 0.6
            assert results[0].page_number == 1
            assert results[0].document_name == "ai_handbook.pdf"
            assert emb_time > 0.0
            print(f"✓ Passed: Dense retrieved top match '{results[0].chunk_id}' with similarity {results[0].dense_score:.4f} in {emb_time:.2f}ms.")
        finally:
            rs_mod.vector_store = orig_vs


def test_bm25_retrieval_standalone():
    """Test 3: BM25 lexical retrieval standalone."""
    print("\n--- Test 3: BM25 Lexical Retrieval Standalone ---")
    with SafeTempDir() as tmp_bm:
        bm = BM25IndexService(index_dir=tmp_bm)
        chunks = [
            ChunkMetadata(
                chunk_id="chunk_tcp",
                document_id="doc_net",
                document_name="protocols.pdf",
                page_number=3,
                chunk_index=0,
                chunk_text="Transmission Control Protocol ensures in-order delivery through three-way SYN-ACK handshakes.",
                token_count=14,
                character_count=95,
            ),
            ChunkMetadata(
                chunk_id="chunk_udp",
                document_id="doc_net",
                document_name="protocols.pdf",
                page_number=4,
                chunk_index=1,
                chunk_text="User Datagram Protocol provides low latency packet transmission with zero connection state.",
                token_count=13,
                character_count=92,
            ),
        ]
        bm.add_chunks(chunks)

        service = RetrievalService()
        from app.services import retrieval_service as rs_mod
        orig_bm = rs_mod.bm25_service
        rs_mod.bm25_service = bm

        try:
            results, count = service.retrieve_bm25("SYN-ACK handshake in TCP", top_k=2)
            assert len(results) > 0
            assert results[0].chunk_id == "chunk_tcp"
            assert results[0].bm25_rank == 1
            assert results[0].bm25_score is not None and results[0].bm25_score > 0.0
            assert results[0].page_number == 3
            assert results[0].document_name == "protocols.pdf"
            print(f"✓ Passed: BM25 retrieved keyword match '{results[0].chunk_id}' with score {results[0].bm25_score:.4f}.")
        finally:
            rs_mod.bm25_service = orig_bm


def test_hybrid_rrf_rank_fusion_and_deduplication():
    """Test 4 & 5: Hybrid RRF rank fusion and deduplication."""
    print("\n--- Test 4 & 5: Hybrid RRF Fusion & Deduplication ---")
    with SafeTempDir() as tmp_chroma, SafeTempDir() as tmp_bm:
        vs = ChromaVectorStore(persist_directory=tmp_chroma)
        bm = BM25IndexService(index_dir=tmp_bm)

        chunks = [
            ChunkMetadata(
                chunk_id="chunk_common",
                document_id="doc_1",
                document_name="shared.pdf",
                page_number=1,
                chunk_index=0,
                chunk_text="FastAPI framework provides high performance async REST API endpoints in Python.",
                token_count=12,
                character_count=80,
            ),
            ChunkMetadata(
                chunk_id="chunk_dense_only",
                document_id="doc_1",
                document_name="shared.pdf",
                page_number=2,
                chunk_index=1,
                chunk_text="Starlette toolkit provides ASGI routing and web server gateways.",
                token_count=11,
                character_count=67,
            ),
            ChunkMetadata(
                chunk_id="chunk_bm25_only",
                document_id="doc_1",
                document_name="shared.pdf",
                page_number=3,
                chunk_index=2,
                chunk_text="Uvicorn uses uvloop and httptools for extreme concurrency in ASGI applications.",
                token_count=12,
                character_count=79,
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
            results, emb_time, d_count, b_count = service.retrieve_hybrid_rrf(
                "FastAPI async REST endpoints", top_k=3, rrf_k=60
            )

            # Deduplication check: each chunk_id must appear at most once
            chunk_ids = [r.chunk_id for r in results]
            assert len(chunk_ids) == len(set(chunk_ids)), f"Duplicate chunks found in hybrid output: {chunk_ids}"

            # Rank check: chunk_common matches both dense and BM25 keywords, so it should have the highest RRF score
            top = results[0]
            assert top.chunk_id == "chunk_common"
            assert top.dense_rank is not None
            assert top.bm25_rank is not None
            assert top.rrf_score is not None

            # Verify RRF score matches the mathematical formula exactly
            expected_score = round((1.0 / (60 + top.dense_rank)) + (1.0 / (60 + top.bm25_rank)), 6)
            assert top.rrf_score == expected_score, f"RRF mismatch: {top.rrf_score} != {expected_score}"
            print(f"✓ Passed: Fused 3 candidates into {len(results)} unique results. Top chunk '{top.chunk_id}' scored {top.rrf_score:.6f}.")
        finally:
            rs_mod.vector_store = orig_vs
            rs_mod.bm25_service = orig_bm


def test_missing_results_in_one_retrieval_method():
    """Test 6: Handling chunks appearing in only one retrieval method."""
    print("\n--- Test 6: Missing Results in One Retrieval Method ---")
    with SafeTempDir() as tmp_chroma, SafeTempDir() as tmp_bm:
        vs = ChromaVectorStore(persist_directory=tmp_chroma)
        bm = BM25IndexService(index_dir=tmp_bm)

        # Chunk A is in both
        chunk_a = ChunkMetadata(
            chunk_id="chunk_a",
            document_id="doc_x",
            document_name="x.pdf",
            page_number=1,
            chunk_index=0,
            chunk_text="Database indexing speeds up SQL query lookups via B-tree data structures.",
            token_count=13,
            character_count=76,
        )
        # Chunk B is ONLY in BM25
        chunk_b = ChunkMetadata(
            chunk_id="chunk_b",
            document_id="doc_x",
            document_name="x.pdf",
            page_number=2,
            chunk_index=1,
            chunk_text="PostgreSQL specific syntax and schema migrations with alembic.",
            token_count=9,
            character_count=63,
        )

        vs.add_chunks([chunk_a])
        bm.add_chunks([chunk_a, chunk_b])

        service = RetrievalService()
        from app.services import retrieval_service as rs_mod
        orig_vs = rs_mod.vector_store
        orig_bm = rs_mod.bm25_service
        rs_mod.vector_store = vs
        rs_mod.bm25_service = bm

        try:
            results, _, _, _ = service.retrieve_hybrid_rrf("schema migrations with alembic", top_k=5, rrf_k=60)
            b_result = next((r for r in results if r.chunk_id == "chunk_b"), None)
            assert b_result is not None, "chunk_b should have been retrieved by BM25"
            assert b_result.dense_rank is None
            assert b_result.dense_score is None
            assert b_result.bm25_rank is not None
            assert b_result.bm25_score is not None

            # Its RRF score must come strictly from BM25 rank: 1 / (60 + bm25_rank)
            expected_rrf = round(1.0 / (60 + b_result.bm25_rank), 6)
            assert b_result.rrf_score == expected_rrf
            print(f"✓ Passed: Single-source chunk correctly has dense_rank=None and single-rank RRF score ({b_result.rrf_score}).")
        finally:
            rs_mod.vector_store = orig_vs
            rs_mod.bm25_service = orig_bm


def test_stable_deterministic_ranking():
    """Test 7: Stable deterministic ranking across repeated executions."""
    print("\n--- Test 7: Stable Deterministic Ranking ---")
    with SafeTempDir() as tmp_chroma, SafeTempDir() as tmp_bm:
        vs = ChromaVectorStore(persist_directory=tmp_chroma)
        bm = BM25IndexService(index_dir=tmp_bm)

        chunks = [
            ChunkMetadata(
                chunk_id=f"det_chunk_{i}",
                document_id="doc_det",
                document_name="deterministic.pdf",
                page_number=i + 1,
                chunk_index=i,
                chunk_text=f"Paragraph {i} describing operating system kernels, scheduling, and memory paging.",
                token_count=13,
                character_count=78,
            )
            for i in range(5)
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
            req = RetrievalRequest(query="operating system memory paging", retrieval_method=RetrievalMethod.HYBRID, top_k=3)
            resp1 = service.search(req)
            resp2 = service.search(req)

            assert len(resp1.results) == len(resp2.results)
            for r1, r2 in zip(resp1.results, resp2.results):
                assert r1.chunk_id == r2.chunk_id
                assert r1.score == r2.score
                assert r1.rank == r2.rank
            print("✓ Passed: Repeated queries returned identical rankings and scores.")
        finally:
            rs_mod.vector_store = orig_vs
            rs_mod.bm25_service = orig_bm


def test_page_metadata_preservation():
    """Test 8: Original page numbers and document names preserved in retrieved chunks."""
    print("\n--- Test 8: Page Metadata Preservation ---")
    with SafeTempDir() as tmp_chroma, SafeTempDir() as tmp_bm:
        vs = ChromaVectorStore(persist_directory=tmp_chroma)
        bm = BM25IndexService(index_dir=tmp_bm)

        chunk = ChunkMetadata(
            chunk_id="doc_xyz_p42_c7",
            document_id="doc_xyz",
            document_name="quantum_computing.pdf",
            page_number=42,
            chunk_index=7,
            chunk_text="Qubits exhibit quantum superposition and entanglement for exponential speedups.",
            token_count=11,
            character_count=80,
        )
        vs.add_chunks([chunk])
        bm.add_chunks([chunk])

        service = RetrievalService()
        from app.services import retrieval_service as rs_mod
        orig_vs = rs_mod.vector_store
        orig_bm = rs_mod.bm25_service
        rs_mod.vector_store = vs
        rs_mod.bm25_service = bm

        try:
            for method in [RetrievalMethod.BM25, RetrievalMethod.DENSE, RetrievalMethod.HYBRID]:
                req = RetrievalRequest(query="qubits quantum superposition", retrieval_method=method, top_k=1)
                resp = service.search(req)
                assert len(resp.results) == 1
                c = resp.results[0]
                assert c.chunk_id == "doc_xyz_p42_c7"
                assert c.page_number == 42
                assert c.document_name == "quantum_computing.pdf"
                assert c.chunk_index == 7
            print("✓ Passed: Page number 42 and document name verified across all 3 retrieval methods.")
        finally:
            rs_mod.vector_store = orig_vs
            rs_mod.bm25_service = orig_bm


def test_empty_queries_and_empty_indices():
    """Test 9: Graceful handling of empty queries and empty indices."""
    print("\n--- Test 9: Empty Queries & Empty Indices ---")
    with SafeTempDir() as empty_chroma, SafeTempDir() as empty_bm:
        vs = ChromaVectorStore(persist_directory=empty_chroma)
        bm = BM25IndexService(index_dir=empty_bm)

        service = RetrievalService()
        from app.services import retrieval_service as rs_mod
        orig_vs = rs_mod.vector_store
        orig_bm = rs_mod.bm25_service
        rs_mod.vector_store = vs
        rs_mod.bm25_service = bm

        try:
            # 1. Empty string query
            req_empty = RetrievalRequest(query="   ", retrieval_method=RetrievalMethod.HYBRID)
            resp_empty = service.search(req_empty)
            assert resp_empty.total_results == 0
            assert resp_empty.results == []

            # 2. Non-empty query on empty index
            req_valid = RetrievalRequest(query="search something", retrieval_method=RetrievalMethod.HYBRID)
            resp_valid = service.search(req_valid)
            assert resp_valid.total_results == 0
            assert resp_valid.results == []
            print("✓ Passed: Empty queries and queries on empty index returned 0 results cleanly without errors.")
        finally:
            rs_mod.vector_store = orig_vs
            rs_mod.bm25_service = orig_bm


def test_api_route_controller_validation():
    """Test 10: FastAPI route controller validation for POST /api/v1/retrieval/search."""
    print("\n--- Test 10: API Route Controller Validation ---")
    # Test through the controller function
    req = RetrievalRequest(
        query="retrieval architecture",
        retrieval_method=RetrievalMethod.HYBRID,
        top_k=3,
        rrf_k=60,
    )
    resp = search_documents(req)
    assert isinstance(resp, RetrievalResponse)
    assert resp.query == "retrieval architecture"
    assert resp.retrieval_method == "hybrid"
    assert resp.execution_time_ms >= 0.0
    assert resp.rrf_k == 60
    print(f"✓ Passed: Controller returned valid RetrievalResponse (execution time: {resp.execution_time_ms:.2f} ms).")


def test_retrieval_timing_isolation():
    """Test 11: Verification that embedding time and total retrieval time are separately tracked."""
    print("\n--- Test 11: Retrieval Timing Isolation ---")
    service = RetrievalService()

    # 1. BM25 should have embedding_time_ms == None
    bm_resp = service.search(RetrievalRequest(query="test", retrieval_method=RetrievalMethod.BM25))
    assert bm_resp.embedding_time_ms is None
    assert bm_resp.execution_time_ms >= 0.0

    # 2. Dense should have embedding_time_ms as a positive float
    dense_resp = service.search(RetrievalRequest(query="test", retrieval_method=RetrievalMethod.DENSE))
    assert dense_resp.embedding_time_ms is not None
    assert dense_resp.embedding_time_ms >= 0.0
    assert dense_resp.execution_time_ms >= dense_resp.embedding_time_ms

    # 3. Hybrid should have embedding_time_ms as a positive float
    hybrid_resp = service.search(RetrievalRequest(query="test", retrieval_method=RetrievalMethod.HYBRID))
    assert hybrid_resp.embedding_time_ms is not None
    assert hybrid_resp.execution_time_ms >= hybrid_resp.embedding_time_ms
    print(f"✓ Passed: Timings correctly isolated (Dense emb: {dense_resp.embedding_time_ms} ms, total: {dense_resp.execution_time_ms} ms; BM25 emb: {bm_resp.embedding_time_ms}).")


def run_all_tests():
    print("=" * 65)
    print("STARTING PHASE 3 TEST SUITE: HYBRID RETRIEVAL & RRF")
    print("=" * 65)
    test_rrf_mathematical_formula_verification()
    test_dense_retrieval_standalone()
    test_bm25_retrieval_standalone()
    test_hybrid_rrf_rank_fusion_and_deduplication()
    test_missing_results_in_one_retrieval_method()
    test_stable_deterministic_ranking()
    test_page_metadata_preservation()
    test_empty_queries_and_empty_indices()
    test_api_route_controller_validation()
    test_retrieval_timing_isolation()
    print("\n" + "=" * 65)
    print("ALL PHASE 3 TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
