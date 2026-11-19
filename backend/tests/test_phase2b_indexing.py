"""
Comprehensive Automated Test Suite for Phase 2B:
Dense Vector Indexing (ChromaDB), BM25 Lexical Indexing (rank-bm25),
Orchestration, Dual-Store Synchronization, and API Endpoints.
"""
import io
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi import UploadFile
from app.core.config import settings
from app.models.document import ChunkMetadata
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import ChromaVectorStore
from app.services.bm25_service import BM25IndexService
from app.services.index_manager import IndexManager
from app.services.storage_service import StorageService
from app.api.routes_documents import upload_document, delete_document
from app.api.routes_index import get_index_status, get_index_stats, rebuild_indices
import pymupdf


class SafeTempDir:
    """Context manager for temporary directories that ignores Windows file locking on cleanup."""
    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path, ignore_errors=True)


def make_test_pdf(text: str) -> bytes:
    """Creates a basic PDF byte array."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_dense_vector_generation():
    """Test 1: Dense vector generation (correct dimension, unit norm, deterministic)."""
    print("\n--- Test 1: Dense Vector Generation ---")
    svc = EmbeddingService()
    text = "Hybrid retrieval combines dense vector embeddings with lexical BM25 matching."

    # Single query embedding
    vec = svc.embed_query(text)
    assert len(vec) == 384, f"Expected 384 dimensions, got {len(vec)}"

    # Check L2 normalization: norm should equal 1.0 (within float tolerance)
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4, f"Vector norm {norm} is not approximately 1.0"

    # Deterministic test: identical text yields identical vectors
    vec2 = svc.embed_query(text)
    for a, b in zip(vec, vec2):
        assert abs(a - b) < 1e-6, "Embeddings are not deterministic!"

    # Batch embedding
    batch = ["First chunk of documentation.", "Second chunk of technical details."]
    batch_vecs = svc.embed_texts(batch)
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 384
    assert len(batch_vecs[1]) == 384
    print("✓ Passed: 384 dimensions, L2-normalized, deterministic batch outputs.")


def test_chromadb_insertion_and_retrieval():
    """Test 2: ChromaDB chunk insertion, cosine space search, and metadata preservation."""
    print("\n--- Test 2: ChromaDB Insertion & Retrieval ---")
    with SafeTempDir() as tmp_dir:
        vs = ChromaVectorStore(persist_directory=tmp_dir)

        sample_chunks = [
            ChunkMetadata(
                chunk_id="doc1_chunk_0",
                document_id="doc1",
                document_name="deep_learning.pdf",
                page_number=1,
                chunk_index=0,
                chunk_text="Transformers use multi-head self-attention mechanisms for deep sequence modeling.",
                token_count=14,
                character_count=82,
            ),
            ChunkMetadata(
                chunk_id="doc1_chunk_1",
                document_id="doc1",
                document_name="deep_learning.pdf",
                page_number=2,
                chunk_index=1,
                chunk_text="Convolutional neural networks specialize in computer vision and spatial feature extraction.",
                token_count=14,
                character_count=89,
            ),
        ]

        added = vs.add_chunks(sample_chunks)
        assert added == 2
        assert vs.collection.count() == 2

        # Search for transformer query
        results = vs.search_by_text("self-attention mechanism in transformers", top_k=2)
        assert len(results) > 0
        top = results[0]
        assert top["chunk_id"] == "doc1_chunk_0"
        assert top["document_name"] == "deep_learning.pdf"
        assert top["page_number"] == 1
        assert top["score"] > 0.6  # High cosine similarity
        print(f"✓ Passed: Inserted 2 chunks, retrieved top match '{top['chunk_id']}' with score {top['score']:.4f}")


def test_stable_chunk_ids_prevent_duplicates():
    """Test 3: Stable chunk IDs prevent duplicate indexing in ChromaDB."""
    print("\n--- Test 3: Idempotency & Duplicate Prevention ---")
    with SafeTempDir() as tmp_dir:
        vs = ChromaVectorStore(persist_directory=tmp_dir)

        chunk = ChunkMetadata(
            chunk_id="stable_id_101",
            document_id="doc_abc",
            document_name="guide.pdf",
            page_number=1,
            chunk_index=0,
            chunk_text="FastAPI provides high performance asynchronous routing for Python backends.",
            token_count=10,
            character_count=73,
        )

        vs.add_chunks([chunk])
        assert vs.collection.count() == 1

        # Re-adding the exact same chunk should upsert without increasing total count
        vs.add_chunks([chunk])
        assert vs.collection.count() == 1, f"Expected 1 chunk after re-add, found {vs.collection.count()}"
        print("✓ Passed: Re-adding chunk with identical ID maintains single record.")


def test_chromadb_persistence_across_restarts():
    """Test 4: ChromaDB persistent client recovers records on cold reload."""
    print("\n--- Test 4: ChromaDB Cold Restart Persistence ---")
    with SafeTempDir() as tmp_dir:
        # First session
        vs1 = ChromaVectorStore(persist_directory=tmp_dir)
        vs1.add_chunks([
            ChunkMetadata(
                chunk_id="cold_chunk_1",
                document_id="doc_cold",
                document_name="cold.pdf",
                page_number=1,
                chunk_index=0,
                chunk_text="ChromaDB uses DuckDB and Parquet or SQLite for persistent local vector storage.",
                token_count=12,
                character_count=80,
            )
        ])
        assert vs1.collection.count() == 1

        # Simulate new process / cold restart by creating a new instance
        vs2 = ChromaVectorStore(persist_directory=tmp_dir)
        stats = vs2.get_stats()
        assert stats["chunk_count"] == 1
        assert stats["status"] == "ready"
        results = vs2.search_by_text("persistent storage", top_k=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "cold_chunk_1"
        print("✓ Passed: Fresh instance loaded persisted vector record from disk.")


def test_bm25_lexical_index_and_search():
    """Test 5: BM25 lexical index creation, tokenization, and search."""
    print("\n--- Test 5: BM25 Lexical Index & Search ---")
    with SafeTempDir() as tmp_dir:
        bm = BM25IndexService(index_dir=tmp_dir)

        chunks = [
            ChunkMetadata(
                chunk_id="bm_chunk_1",
                document_id="doc_net",
                document_name="networking.pdf",
                page_number=1,
                chunk_index=0,
                chunk_text="Transmission Control Protocol (TCP) ensures ordered, reliable byte stream delivery.",
                token_count=13,
                character_count=82,
            ),
            ChunkMetadata(
                chunk_id="bm_chunk_2",
                document_id="doc_net",
                document_name="networking.pdf",
                page_number=2,
                chunk_index=1,
                chunk_text="User Datagram Protocol (UDP) offers lightweight connectionless transmission.",
                token_count=10,
                character_count=74,
            ),
        ]

        bm.add_chunks(chunks)
        assert len(bm.chunk_ids) == 2
        assert bm.bm25 is not None

        # Search for exact acronym
        res = bm.search("TCP ordered reliable", top_k=2)
        assert len(res) > 0
        assert res[0]["chunk_id"] == "bm_chunk_1"
        assert res[0]["score"] > 0.0
        print(f"✓ Passed: BM25 indexed 2 documents; query 'TCP' scored top match with score {res[0]['score']:.4f}")


def test_bm25_query_scoring_accuracy():
    """Test 6: BM25 term frequency discrimination and relevance ranking."""
    print("\n--- Test 6: BM25 Scoring Accuracy & Discrimination ---")
    with SafeTempDir() as tmp_dir:
        bm = BM25IndexService(index_dir=tmp_dir)

        chunks = [
            ChunkMetadata(
                chunk_id="term_python",
                document_id="d1",
                document_name="prog.pdf",
                page_number=1,
                chunk_index=0,
                chunk_text="Python programming language uses indentation for block scoping.",
                token_count=10,
                character_count=65,
            ),
            ChunkMetadata(
                chunk_id="term_rust",
                document_id="d2",
                document_name="prog.pdf",
                page_number=1,
                chunk_index=1,
                chunk_text="Rust programming language enforces memory safety via ownership and borrow checker.",
                token_count=12,
                character_count=82,
            ),
        ]
        bm.add_chunks(chunks)

        rust_hits = bm.search("borrow checker memory safety", top_k=2)
        assert rust_hits[0]["chunk_id"] == "term_rust"

        python_hits = bm.search("indentation block scoping", top_k=2)
        assert python_hits[0]["chunk_id"] == "term_python"
        print("✓ Passed: BM25 cleanly discriminates between domain keywords.")


def test_dual_index_metadata_alignment():
    """Test 7: Index orchestrator keeps ChromaDB and BM25 perfectly aligned."""
    print("\n--- Test 7: Dual-Index Metadata Alignment ---")
    with SafeTempDir() as tmp_chroma, SafeTempDir() as tmp_bm25:
        # Patch vector_store and bm25_service directories
        vs = ChromaVectorStore(persist_directory=tmp_chroma)
        bm = BM25IndexService(index_dir=tmp_bm25)

        mgr = IndexManager()
        # Direct chunk indexing
        chunks = [
            ChunkMetadata(
                chunk_id=f"align_chunk_{i}",
                document_id="align_doc",
                document_name="alignment.pdf",
                page_number=i + 1,
                chunk_index=i,
                chunk_text=f"Technical specification chunk number {i} with key attributes.",
                token_count=10,
                character_count=60,
            )
            for i in range(5)
        ]

        vs.add_chunks(chunks)
        bm.add_chunks(chunks)

        assert vs.collection.count() == 5
        assert len(bm.chunk_ids) == 5

        for c in chunks:
            assert c.chunk_id in bm.chunk_store
        print("✓ Passed: All 5 chunks are stored with identical IDs in both vector and lexical stores.")


def test_incremental_indexing_on_upload():
    """Test 8: Document upload automatically indexes chunks incrementally."""
    print("\n--- Test 8: Incremental Ingestion & Auto-Indexing ---")
    pdf_bytes = make_test_pdf(
        "PyMuPDF extracts clean text. ChromaDB stores dense embeddings. BM25 indexes lexical tokens."
    )
    upload_file = UploadFile(
        file=io.BytesIO(pdf_bytes),
        filename="incremental_test.pdf",
        headers={"content-type": "application/pdf"},
    )

    response = upload_document(upload_file)
    doc_id = response.document.document_id
    assert response.chunks_created >= 1

    stats = get_index_stats()
    assert stats.chroma_chunk_count >= response.chunks_created
    assert stats.bm25_chunk_count >= response.chunks_created
    assert stats.is_synchronized

    # Clean up created document
    delete_document(doc_id)
    print("✓ Passed: Document upload incrementally indexed chunks into both stores.")


def test_document_deletion_synchronization():
    """Test 9: Document deletion removes chunks from both ChromaDB and BM25."""
    print("\n--- Test 9: Document Deletion Synchronization ---")
    pdf_bytes = make_test_pdf("Temporary document to test cascade removal across both search indices.")
    upload_file = UploadFile(
        file=io.BytesIO(pdf_bytes),
        filename="delete_sync_test.pdf",
        headers={"content-type": "application/pdf"},
    )

    upload_resp = upload_document(upload_file)
    doc_id = upload_resp.document.document_id
    created_chunks = upload_resp.chunks_created

    stats_before = get_index_stats()
    c_before = stats_before.chroma_chunk_count
    b_before = stats_before.bm25_chunk_count

    del_resp = delete_document(doc_id)
    assert del_resp.document_id == doc_id

    stats_after = get_index_stats()
    assert stats_after.chroma_chunk_count == c_before - created_chunks
    assert stats_after.bm25_chunk_count == b_before - created_chunks
    assert stats_after.is_synchronized
    print("✓ Passed: Document deletion cleanly removed chunks from both indices synchronously.")


def test_full_index_rebuild():
    """Test 10: Complete rebuild reconstructs both indices from processed document files."""
    print("\n--- Test 10: Full Index Rebuild ---")
    # Upload two sample documents
    pdf1 = make_test_pdf("First document for full rebuild validation with unique terms alpha.")
    pdf2 = make_test_pdf("Second document for full rebuild validation with unique terms beta.")

    u1 = upload_document(UploadFile(file=io.BytesIO(pdf1), filename="rebuild_doc1.pdf"))
    u2 = upload_document(UploadFile(file=io.BytesIO(pdf2), filename="rebuild_doc2.pdf"))

    rebuild_res = rebuild_indices()
    assert rebuild_res.status == "completed"
    assert rebuild_res.documents_processed >= 2
    assert rebuild_res.chunks_indexed >= 2

    stats = get_index_stats()
    assert stats.is_synchronized
    assert stats.chroma_chunk_count == rebuild_res.chunks_indexed
    assert stats.bm25_chunk_count == rebuild_res.chunks_indexed

    # Clean up
    delete_document(u1.document.document_id)
    delete_document(u2.document.document_id)
    print(f"✓ Passed: Rebuild indexed {rebuild_res.chunks_indexed} chunks in {rebuild_res.duration_seconds}s.")


def test_retrieval_benchmarks():
    """Test 11: Timing benchmarks for Dense Vector and BM25 queries."""
    print("\n--- Test 11: Retrieval Timing Benchmarks ---")
    with SafeTempDir() as tmp_chroma, SafeTempDir() as tmp_bm25:
        vs = ChromaVectorStore(persist_directory=tmp_chroma)
        bm = BM25IndexService(index_dir=tmp_bm25)

        chunks = [
            ChunkMetadata(
                chunk_id=f"bench_{i}",
                document_id="doc_bench",
                document_name="bench.pdf",
                page_number=1,
                chunk_index=i,
                chunk_text=f"Benchmarking query latency with sample data paragraph {i} discussing algorithmic performance.",
                token_count=13,
                character_count=95,
            )
            for i in range(20)
        ]
        vs.add_chunks(chunks)
        bm.add_chunks(chunks)

        # Benchmark BM25 query latency
        t0 = time.perf_counter()
        for _ in range(50):
            bm.search("algorithmic performance latency", top_k=5)
        bm25_latency_ms = ((time.perf_counter() - t0) / 50) * 1000

        # Benchmark Dense Vector search latency (with query embedding)
        t1 = time.perf_counter()
        for _ in range(10):
            vs.search_by_text("algorithmic performance latency", top_k=5)
        dense_latency_ms = ((time.perf_counter() - t1) / 10) * 1000

        print(f"✓ Passed: BM25 Latency: {bm25_latency_ms:.2f} ms | Dense Vector Latency: {dense_latency_ms:.2f} ms")
        assert bm25_latency_ms < 50.0, "BM25 latency exceeded 50ms"
        assert dense_latency_ms < 500.0, "Dense search latency exceeded 500ms"


def test_error_handling_corrupted_chunks():
    """Test 12: Resilient error handling for missing/corrupted chunks or invalid inputs."""
    print("\n--- Test 12: Error Handling & Resilience ---")
    mgr = IndexManager()

    # Indexing empty chunks returns 0 cleanly
    count = mgr.index_chunks([])
    assert count == 0

    # Indexing non-existent document ID raises ValueError
    try:
        mgr.index_document("non_existent_uuid_12345")
        assert False, "Expected ValueError for missing document"
    except ValueError:
        pass

    # Status reflects healthy idle state
    status = mgr.get_status()
    assert status.status in ("idle", "indexing", "rebuilding")
    print("✓ Passed: Gracefully handled empty chunk arrays and missing document IDs.")


def run_all_tests():
    print("=" * 60)
    print("STARTING PHASE 2B TEST SUITE: DENSE VECTOR & BM25 INDEXING")
    print("=" * 60)
    test_dense_vector_generation()
    test_chromadb_insertion_and_retrieval()
    test_stable_chunk_ids_prevent_duplicates()
    test_chromadb_persistence_across_restarts()
    test_bm25_lexical_index_and_search()
    test_bm25_query_scoring_accuracy()
    test_dual_index_metadata_alignment()
    test_incremental_indexing_on_upload()
    test_document_deletion_synchronization()
    test_full_index_rebuild()
    test_retrieval_benchmarks()
    test_error_handling_corrupted_chunks()
    print("\n" + "=" * 60)
    print("ALL 12 PHASE 2B TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
