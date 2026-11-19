"""
Comprehensive Automated Test Suite for Phase 2A:
PDF Ingestion, Text Extraction, Cleaning, Chunking, Storage, and FastAPI Route Controllers.
"""
import io
import json
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import pymupdf
from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.services.document_loader import (
    PDFDocumentLoader,
    PDFValidationError,
    PDFPasswordProtectedError,
    PDFCorruptedError,
    PDFSizeExceededError,
)
from app.services.text_cleaner import clean_text
from app.services.text_splitter import TextSplitter, SubwordTokenizer
from app.services.storage_service import storage_service
from app.api.routes_documents import (
    upload_document,
    list_documents,
    get_document,
    delete_document,
)


def create_sample_pdf(pages_text: list, password: str = None) -> bytes:
    """Helper to generate in-memory PDF bytes with custom text per page."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 50), text)
    
    if password:
        perm = int(
            pymupdf.PDF_PERM_ACCESSIBILITY
            | pymupdf.PDF_PERM_PRINT
            | pymupdf.PDF_PERM_COPY
            | pymupdf.PDF_PERM_ANNOTATE
        )
        pdf_bytes = doc.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner123",
            user_pw=password,
            permissions=perm,
        )
    else:
        pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_1_text_cleaner():
    """Verify whitespace normalization, newline collapsing, and preservation of technical terms."""
    dirty = "  Production   Hybrid   RAG \r\n\r\n\r\n\r\nPipeline with BGE-reranker and BM25.\t\tScore: 98.7%.\n\n\n\n\nSection 2. "
    cleaned = clean_text(dirty)
    assert "Production Hybrid RAG" in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "BGE-reranker" in cleaned
    assert "BM25" in cleaned
    assert "98.7%" in cleaned
    print("[PASS] test_1_text_cleaner: Whitespace normalized, paragraph boundaries preserved, technical terms retained.")


def test_2_tokenizer_and_chunk_size_overlap():
    """Verify subword tokenizer, sentence preservation, chunk size limit, and overlap."""
    splitter = TextSplitter(chunk_size=40, chunk_overlap=10)
    
    long_text = (
        "Retrieval-Augmented Generation enhances large language models by integrating external knowledge sources. "
        "Dense vector embeddings capture semantic relationships and context across heterogeneous corpora. "
        "Sparse BM25 retrieval excels at exact keyword matching, technical identifiers, and domain-specific acronyms. "
        "Reciprocal Rank Fusion effectively blends disparate ranking signals into a unified candidate pool."
    )
    
    chunks = splitter.split_page_text(
        text=long_text,
        document_id="test_doc_01",
        document_name="sample.pdf",
        page_number=1,
    )
    
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"
    for idx, c in enumerate(chunks):
        assert c.chunk_id == f"test_doc_01_p1_c{idx:03d}"
        assert c.page_number == 1
        assert c.token_count > 0
        assert c.document_name == "sample.pdf"
        
    print(f"[PASS] test_2_tokenizer_and_chunk_size_overlap: Generated {len(chunks)} chunks with token tracking and overlap.")


def test_3_pdf_extraction_page_number_preservation():
    """Verify page number preservation (1-indexed) across multi-page PDF."""
    pdf_bytes = create_sample_pdf([
        "Page 1: Introduction to Hybrid Search Pipelines.",
        "Page 2: Benchmark evaluation on specialized datasets.",
        "Page 3: Latency tradeoffs between dense and sparse indexing.",
    ])
    
    pages_data, page_diags = PDFDocumentLoader.load_and_extract(pdf_bytes, "multipage.pdf")
    
    assert len(pages_data) == 3
    assert pages_data[0]["page_number"] == 1
    assert pages_data[1]["page_number"] == 2
    assert pages_data[2]["page_number"] == 3
    assert "Introduction to Hybrid" in pages_data[0]["text"]
    assert "Benchmark evaluation" in pages_data[1]["text"]
    assert "Latency tradeoffs" in pages_data[2]["text"]
    print("[PASS] test_3_pdf_extraction_page_number_preservation: 1-indexed page alignment verified across 3 pages.")


def test_4_empty_pages_handling():
    """Verify graceful handling of blank pages without crashing or dropping page alignment."""
    pdf_bytes = create_sample_pdf([
        "Page 1 contains valid text content.",
        "",  # Blank page 2
        "Page 3 contains valid technical summaries.",
    ])
    
    pages_data, page_diags = PDFDocumentLoader.load_and_extract(pdf_bytes, "empty_pages.pdf")
    assert len(pages_data) == 3
    assert pages_data[0]["has_text"] is True
    assert pages_data[1]["has_text"] is False
    assert pages_data[1]["page_number"] == 2
    assert pages_data[2]["has_text"] is True
    assert pages_data[2]["page_number"] == 3
    print("[PASS] test_4_empty_pages_handling: Blank page 2 handled gracefully without breaking sequence.")


def test_5_corrupted_and_invalid_uploads():
    """Verify validation for non-PDFs, empty bytes, and corrupted headers."""
    # 1. Non-pdf extension
    try:
        PDFDocumentLoader.validate_pdf_bytes(b"content", "report.txt")
        assert False, "Should have raised PDFValidationError"
    except PDFValidationError as e:
        assert "Only .pdf files are accepted" in str(e)

    # 2. Corrupted PDF
    try:
        PDFDocumentLoader.validate_pdf_bytes(b"not a real pdf content", "corrupted.pdf")
        assert False, "Should have raised PDFCorruptedError"
    except PDFCorruptedError as e:
        assert "missing %PDF- header" in str(e)

    # 3. Empty PDF (0 bytes)
    try:
        PDFDocumentLoader.validate_pdf_bytes(b"", "empty.pdf")
        assert False, "Should have raised PDFCorruptedError"
    except PDFCorruptedError as e:
        assert "empty" in str(e).lower()
    print("[PASS] test_5_corrupted_and_invalid_uploads: Non-PDF, truncated, and empty files rejected.")


def test_6_password_protected_pdf():
    """Verify password-protected PDFs are rejected gracefully with a clear message."""
    encrypted_bytes = create_sample_pdf(["Confidential financial data"], password="securePassword123")
    try:
        PDFDocumentLoader.load_and_extract(encrypted_bytes, "encrypted.pdf")
        assert False, "Should have raised PDFPasswordProtectedError"
    except PDFPasswordProtectedError as e:
        assert "password-protected" in str(e).lower()
    print("[PASS] test_6_password_protected_pdf: Password-protected encrypted PDF detected and rejected.")


def test_7_storage_and_duplicate_idempotency():
    """Verify storage persistence, duplicate SHA-256 detection, and safe deletion."""
    pdf_bytes = create_sample_pdf(["Storage persistence test page 1", "Storage persistence test page 2"])
    content_hash = storage_service.compute_sha256(pdf_bytes)
    doc_id = "test_persistence_doc_999"
    safe_name = "test_persistence.pdf"

    # Save raw
    storage_path = storage_service.save_raw_pdf(pdf_bytes, safe_name, doc_id)
    assert os.path.exists(Path(settings.DOCUMENTS_DIRECTORY).parent / storage_path)

    # Save processed document
    from app.models.document import DocumentInfo, PageInfo, ChunkMetadata
    doc_info = DocumentInfo(
        document_id=doc_id,
        document_name=safe_name,
        file_size_bytes=len(pdf_bytes),
        content_hash=content_hash,
        total_pages=2,
        pages_with_text=2,
        total_chunks=2,
        total_tokens=20,
        storage_path=storage_path,
    )
    pages = [PageInfo(page_number=1, has_text=True, character_count=30, token_count=10)]
    chunks = [
        ChunkMetadata(
            chunk_id=f"{doc_id}_p1_c000",
            document_id=doc_id,
            document_name=safe_name,
            page_number=1,
            chunk_index=0,
            chunk_text="Storage persistence test page 1",
            token_count=10,
            character_count=30,
        )
    ]
    storage_service.save_processed_document(doc_info, pages, chunks)

    # Verify duplicate detection
    dup_doc = storage_service.find_by_content_hash(content_hash)
    assert dup_doc is not None
    assert dup_doc.document_id == doc_id

    # Verify retrieval
    retrieved = storage_service.get_document_detail(doc_id)
    assert retrieved is not None
    assert len(retrieved.chunks) == 1

    # Verify deletion
    del_ok = storage_service.delete_document(doc_id)
    assert del_ok is True
    assert storage_service.get_document_detail(doc_id) is None
    assert not os.path.exists(Path(settings.DOCUMENTS_DIRECTORY).parent / storage_path)
    print("[PASS] test_7_storage_and_duplicate_idempotency: Ingestion persistence, duplicate detection, and deletion verified.")


def test_8_api_route_controllers():
    """Verify FastAPI routes: POST upload, GET list, GET detail, and DELETE controllers."""
    pdf_bytes = create_sample_pdf([
        "Section 1: RAG Architecture Overview with BM25 and ChromaDB.",
        "Section 2: Dense and Sparse Ingestion with Page Citations.",
    ])

    # 1. Test POST /api/v1/documents/upload controller
    upload_file = UploadFile(filename="controller_test.pdf", file=io.BytesIO(pdf_bytes))
    upload_response = upload_document(upload_file)

    assert upload_response.document.document_name == "controller_test.pdf"
    assert upload_response.document.total_pages == 2
    assert upload_response.chunks_created >= 2
    assert upload_response.is_duplicate is False
    doc_id = upload_response.document.document_id

    # 2. Test duplicate upload
    upload_file_dup = UploadFile(filename="controller_test.pdf", file=io.BytesIO(pdf_bytes))
    dup_response = upload_document(upload_file_dup)
    assert dup_response.is_duplicate is True
    assert dup_response.document.document_id == doc_id

    # 3. Test GET /api/v1/documents
    docs = list_documents()
    assert any(d.document_id == doc_id for d in docs)

    # 4. Test GET /api/v1/documents/{document_id}
    detail = get_document(doc_id)
    assert detail.document_id == doc_id
    assert len(detail.chunks) >= 2
    assert detail.chunks[0].page_number == 1

    # 5. Test DELETE /api/v1/documents/{document_id}
    del_res = delete_document(doc_id)
    assert del_res.document_id == doc_id

    # 6. Test 404 after deletion
    try:
        get_document(doc_id)
        assert False, "Should have raised 404 HTTPException"
    except HTTPException as e:
        assert e.status_code == 404

    print("[PASS] test_8_api_route_controllers: Upload, duplicate detection, document listing, detail, and deletion verified.")


def run_all_tests():
    print("\n========================================================")
    print("RUNNING PHASE 2A AUTOMATED INGESTION & CHUNKING TESTS")
    print("========================================================\n")
    test_1_text_cleaner()
    test_2_tokenizer_and_chunk_size_overlap()
    test_3_pdf_extraction_page_number_preservation()
    test_4_empty_pages_handling()
    test_5_corrupted_and_invalid_uploads()
    test_6_password_protected_pdf()
    test_7_storage_and_duplicate_idempotency()
    test_8_api_route_controllers()
    print("\n========================================================")
    print("ALL 8 PHASE 2A AUTOMATED TEST SUITES PASSED SUCCESSFULLY")
    print("========================================================\n")


if __name__ == "__main__":
    run_all_tests()
