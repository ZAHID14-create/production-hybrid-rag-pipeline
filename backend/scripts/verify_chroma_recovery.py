"""
Verification Script for ChromaDB Persistent Collection Error Recovery
Tests:
1. Backend startup & collection validation
2. Collection existence & actual chunk count
3. Dense retrieval
4. BM25 retrieval
5. Hybrid RRF retrieval
6. BGE reranking
7. Upload a PDF and verify retrieval
8. Delete test PDF and verify synchronized index cleanup
9. Restart backend & verify retrieval continuity
10. Self-healing resilience against stale collection UUIDs
"""
import io
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Alias regex for transformers
import re
sys.modules['regex'] = re

import pymupdf
from fastapi import UploadFile

from app.core.config import settings
from app.models.retrieval import RetrievalMethod, RetrievalRequest
from app.services.retrieval_service import retrieval_service
from app.services.vector_store import vector_store, ChromaVectorStore, COLLECTION_NAME
from app.services.bm25_service import bm25_service
from app.services.index_manager import index_manager
from app.api.routes_documents import upload_document, delete_document


def create_sample_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def run_tests():
    print("==================================================================")
    print("CHROMADB COLLECTION RESILIENCE & RECOVERY VERIFICATION SUITE")
    print("==================================================================")

    # 1 & 2. Verify collection existence and chunk count
    print("\n--- 1 & 2. Testing Collection Existence & Chunk Count ---")
    stats = vector_store.get_stats()
    print("ChromaDB Stats:", stats)
    assert stats["status"] == "ready", f"Expected 'ready', got {stats['status']}"
    assert stats["collection_name"] == COLLECTION_NAME
    assert stats["chunk_count"] > 0, "Expected non-zero chunks in ChromaDB"
    initial_chunks = stats["chunk_count"]
    print(f"[PASS] Collection '{COLLECTION_NAME}' active with {initial_chunks} chunks.")

    # 3. Test Dense Retrieval
    print("\n--- 3. Testing Dense Retrieval ---")
    dense_chunks, emb_time, cand_cnt = retrieval_service.retrieve_dense(
        "Transformer self-attention mechanism", top_k=3
    )
    assert len(dense_chunks) > 0, "Dense retrieval returned 0 chunks"
    print(f"[PASS] Dense retrieved {len(dense_chunks)} chunks (Embedding latency: {emb_time:.2f}ms).")
    for c in dense_chunks:
        print(f"       [{c.score:.4f}] {c.document_name} (p.{c.page_number}): {c.chunk_id}")

    # 4. Test BM25 Retrieval
    print("\n--- 4. Testing BM25 Retrieval ---")
    bm25_chunks, cand_cnt = retrieval_service.retrieve_bm25(
        "Transformer self-attention mechanism", top_k=3
    )
    assert len(bm25_chunks) > 0, "BM25 retrieval returned 0 chunks"
    print(f"[PASS] BM25 retrieved {len(bm25_chunks)} chunks.")
    for c in bm25_chunks:
        print(f"       [{c.score:.4f}] {c.document_name} (p.{c.page_number}): {c.chunk_id}")

    # 5. Test Hybrid RRF Retrieval
    print("\n--- 5. Testing Hybrid RRF Retrieval ---")
    hybrid_resp = retrieval_service.search(RetrievalRequest(
        query="Transformer self-attention mechanism",
        retrieval_method=RetrievalMethod.HYBRID,
        top_k=3,
        rerank=False,
    ))
    assert hybrid_resp.total_results > 0, "Hybrid RRF returned 0 results"
    print(f"[PASS] Hybrid RRF returned {hybrid_resp.total_results} chunks (Round-trip: {hybrid_resp.execution_time_ms:.2f}ms).")
    for c in hybrid_resp.results:
        print(f"       [RRF: {c.score:.4f}] {c.document_name} (p.{c.page_number}): {c.chunk_id}")

    # 6. Test BGE Reranking
    print("\n--- 6. Testing BGE Cross-Encoder Reranking ---")
    rerank_resp = retrieval_service.search(RetrievalRequest(
        query="Transformer self-attention mechanism",
        retrieval_method=RetrievalMethod.HYBRID,
        top_k=5,
        rerank=True,
        reranker_top_k=3,
    ))
    assert rerank_resp.rerank_enabled is True
    assert len(rerank_resp.results) > 0
    top_score = rerank_resp.results[0].score
    print(f"[PASS] BGE Reranking returned {len(rerank_resp.results)} chunks (Rerank time: {rerank_resp.reranking_time_ms:.2f}ms).")
    print(f"       Top rerank score: {top_score:.4f}")
    assert top_score > 0.8, f"Expected top score > 0.8, got {top_score}"

    # 7. Upload a PDF and verify retrieval
    print("\n--- 7. Testing Upload & Retrieval Integration ---")
    unique_phrase = "QuantumCryptographicLatticeKeyExchangeProtocol2026"
    test_pdf_content = (
        f"Chapter on Cryptography. The {unique_phrase} ensures post-quantum security "
        "by relying on the hardness of shortest vector problems over high-dimensional Euclidean lattices."
    )
    pdf_bytes = create_sample_pdf(test_pdf_content)
    test_upload = UploadFile(filename="temp_crypto_test.pdf", file=io.BytesIO(pdf_bytes))
    upload_res = upload_document(test_upload)
    test_doc_id = upload_res.document.document_id
    print(f"       Ingested test document: {test_doc_id} ({upload_res.chunks_created} chunks)")

    # Verify retrieval finds the newly ingested document
    uploaded_search = retrieval_service.search(RetrievalRequest(
        query=unique_phrase,
        retrieval_method=RetrievalMethod.HYBRID,
        top_k=3,
        rerank=False,
    ))
    assert uploaded_search.total_results > 0
    found_doc_ids = [r.document_id for r in uploaded_search.results]
    assert test_doc_id in found_doc_ids, f"Uploaded document {test_doc_id} not found in retrieval results"
    print(f"[PASS] Uploaded PDF successfully retrieved via query '{unique_phrase}'.")

    # 8. Delete test PDF and verify synchronized index cleanup
    print("\n--- 8. Testing Deletion & Synchronized Cleanup ---")
    del_res = delete_document(test_doc_id)
    assert del_res.document_id == test_doc_id

    # Verify chunks disappeared from both ChromaDB and BM25
    deleted_search = retrieval_service.search(RetrievalRequest(
        query=unique_phrase,
        retrieval_method=RetrievalMethod.HYBRID,
        top_k=3,
        rerank=False,
    ))
    found_after_del = [r.document_id for r in deleted_search.results if r.document_id == test_doc_id]
    assert len(found_after_del) == 0, "Deleted document chunks still retrievable!"
    print("[PASS] Test PDF chunks cleanly eliminated from both ChromaDB and BM25.")

    # 9. Restart backend & verify retrieval continuity
    print("\n--- 9. Testing Backend Restart & Continuity ---")
    # Simulate restart by instantiating a fresh ChromaVectorStore
    fresh_store = ChromaVectorStore()
    col = fresh_store.collection
    fresh_count = col.count()
    assert fresh_count == initial_chunks, f"Expected {initial_chunks} chunks after restart, got {fresh_count}"
    print(f"[PASS] Fresh vector store reconnected to collection '{col.name}' ({fresh_count} chunks).")

    # 10. Test Self-Healing Resilience against Stale Collection UUIDs
    print("\n--- 10. Testing Self-Healing against Stale Collection UUID ---")
    # Intentionally corrupt the cached collection ID to simulate an invalid/stale UUID
    class StaleCollectionMock:
        id = "758af473-1c00-4c0d-9f71-39fb35384e9f"
        name = "rag_chunks"
        def count(self):
            raise ValueError(f"Collection [{self.id}] does not exist.")
        def query(self, **kwargs):
            raise ValueError(f"Collection [{self.id}] does not exist.")

    fresh_store._collection = StaleCollectionMock()
    # Now call collection property - it must detect the stale ID and heal itself
    recovered_col = fresh_store.collection
    assert recovered_col.id != "758af473-1c00-4c0d-9f71-39fb35384e9f"
    assert recovered_col.count() == initial_chunks
    print(f"[PASS] Self-healing mechanism successfully recovered from stale UUID [{StaleCollectionMock.id}] -> Active UUID [{recovered_col.id}].")

    print("\n==================================================================")
    print("ALL 10 TESTS PASSED: CHROMADB RECOVERY & RESILIENCE FULLY VERIFIED")
    print("==================================================================")


if __name__ == "__main__":
    run_tests()
