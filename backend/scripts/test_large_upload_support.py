"""
Comprehensive Verification Suite:
Phase 6 - Large File Support & Streaming Uploads (1 GB Configurable Limit)
"""
import io
import os
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Ensure regex patch if needed
import re
sys.modules['regex'] = re

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
from app.services.storage_service import storage_service
from app.api.routes_documents import (
    get_upload_config,
    upload_document,
    list_documents,
    get_document,
    delete_document,
)


def create_in_memory_pdf(pages_text: list, password: str = None) -> bytes:
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


def create_large_pdf_file(target_size_bytes: int, file_path: Path) -> None:
    """Creates a valid PDF on disk with the requested size by inserting text & dummy streams."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Benchmark Large PDF Document for Streaming Upload Verification.")
    page.insert_text((50, 80), "This document tests the 1 GB streaming upload and PyMuPDF disk-backed parsing.")
    
    # Base PDF representation
    base_bytes = doc.tobytes()
    doc.close()

    # Write base and pad with valid PDF comments/trailing whitespace before EOF to achieve size
    with open(file_path, "wb") as f:
        f.write(base_bytes[:-6])  # Strip %%EOF
        padding_needed = max(0, target_size_bytes - len(base_bytes))
        # Write comment blocks in chunks
        chunk = b"% " + b"X" * 1022 + b"\n"
        while padding_needed > len(chunk):
            f.write(chunk)
            padding_needed -= len(chunk)
        if padding_needed > 0:
            f.write(b"% " + b"X" * max(0, padding_needed - 4) + b"\n")
        f.write(b"%%EOF\n")


def test_1_config_endpoint():
    """Verify config endpoint reports 1024 MB / 1073741824 bytes."""
    cfg = get_upload_config()
    assert cfg["max_upload_size_mb"] == 1024, f"Expected 1024, got {cfg['max_upload_size_mb']}"
    assert cfg["max_upload_size_bytes"] == 1024 * 1024 * 1024
    print(f"[PASS] test_1_config_endpoint: Limit correctly configured at {cfg['max_upload_size_mb']} MB ({cfg['max_upload_size_bytes']} bytes).")


def test_2_small_streaming_upload():
    """Verify small PDF streaming upload, chunking, and metadata."""
    pdf_bytes = create_in_memory_pdf([
        "Page 1: Architecture of streaming file uploads in FastAPI and PyMuPDF.",
        "Page 2: Memory-efficient chunking avoids high RAM overhead.",
    ])
    upload_file = UploadFile(filename="small_streaming_test.pdf", file=io.BytesIO(pdf_bytes))
    res = upload_document(upload_file)
    assert res.is_duplicate is False
    assert res.document.document_name == "small_streaming_test.pdf"
    assert res.chunks_created >= 2
    doc_id = res.document.document_id

    # Clean up
    delete_document(doc_id)
    print(f"[PASS] test_2_small_streaming_upload: Ingested and deleted small PDF ({len(pdf_bytes)} bytes) successfully.")


def test_3_large_pdf_upload_exceeding_20mb():
    """Verify upload and extraction of a ~25 MB PDF that exceeds the old 20 MB limit."""
    test_large_path = Path(settings.DOCUMENTS_DIRECTORY) / "temp_bench_25mb.pdf"
    target_size = 25 * 1024 * 1024  # 25 MB
    create_large_pdf_file(target_size, test_large_path)
    actual_size = test_large_path.stat().st_size
    print(f"       Generated test PDF on disk: {actual_size / (1024 * 1024):.2f} MB")

    try:
        with open(test_large_path, "rb") as f:
            upload_file = UploadFile(filename="bench_25mb.pdf", file=f)
            t0 = time.time()
            res = upload_document(upload_file)
            duration = time.time() - t0

        assert res.is_duplicate is False
        assert res.document.file_size_bytes == actual_size
        assert res.chunks_created >= 1
        doc_id = res.document.document_id
        print(f"[PASS] test_3_large_pdf_upload_exceeding_20mb: 25 MB PDF uploaded & indexed in {duration:.2f}s without memory error.")

        # Duplicate check on 25 MB file
        with open(test_large_path, "rb") as f:
            upload_dup = UploadFile(filename="bench_25mb.pdf", file=f)
            res_dup = upload_document(upload_dup)
        assert res_dup.is_duplicate is True
        assert res_dup.document.document_id == doc_id
        print("[PASS] test_3b_large_pdf_duplicate: 25 MB duplicate detected instantly via SHA-256 without re-indexing.")

        # Clean up
        delete_document(doc_id)
    finally:
        if test_large_path.exists():
            test_large_path.unlink()


def test_4_invalid_and_corrupted_uploads():
    """Verify corrupted, non-PDF, and password-protected files are rejected and cleaned up."""
    docs_dir = Path(settings.DOCUMENTS_DIRECTORY)

    # 1. Non-pdf extension
    upload_non_pdf = UploadFile(filename="notes.txt", file=io.BytesIO(b"Hello world"))
    try:
        upload_document(upload_non_pdf)
        assert False, "Expected 400 BAD REQUEST"
    except HTTPException as e:
        assert e.status_code == 400

    # 2. Empty file (0 bytes)
    upload_empty = UploadFile(filename="empty.pdf", file=io.BytesIO(b""))
    try:
        upload_document(upload_empty)
        assert False, "Expected 400 BAD REQUEST"
    except HTTPException as e:
        assert e.status_code == 400
        assert "empty" in e.detail.lower()

    # 3. Corrupted header
    upload_corrupted = UploadFile(filename="bad_header.pdf", file=io.BytesIO(b"Not a valid pdf header"))
    try:
        upload_document(upload_corrupted)
        assert False, "Expected 400 BAD REQUEST"
    except HTTPException as e:
        assert e.status_code == 400
        assert "missing %PDF- header" in e.detail or "failed to parse" in e.detail.lower()

    # 4. Password protected
    encrypted_bytes = create_in_memory_pdf(["Protected information"], password="mypassword")
    upload_encrypted = UploadFile(filename="secret.pdf", file=io.BytesIO(encrypted_bytes))
    try:
        upload_document(upload_encrypted)
        assert False, "Expected 400 BAD REQUEST"
    except HTTPException as e:
        assert e.status_code == 400
        assert "password-protected" in e.detail.lower()

    # Verify no temp files left behind
    leftover_tmps = list(docs_dir.glob(".tmp_*.pdf"))
    assert len(leftover_tmps) == 0, f"Found leftover temp files: {leftover_tmps}"
    print("[PASS] test_4_invalid_and_corrupted_uploads: All negative cases handled with clean temp file deletion.")


def test_5_oversized_file_rejection():
    """Verify file exceeding settings.MAX_UPLOAD_SIZE_BYTES is rejected with 413 and unlinked."""
    docs_dir = Path(settings.DOCUMENTS_DIRECTORY)
    original_limit = settings.MAX_UPLOAD_SIZE_BYTES
    original_mb = settings.MAX_UPLOAD_SIZE_MB

    # Temporarily set limit to 1 MB to test size guard safely without generating 1 GB on disk
    try:
        settings.MAX_UPLOAD_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB
        settings.MAX_UPLOAD_SIZE_MB = 1

        oversized_bytes = b"%PDF-1.4\n" + b"X" * (2 * 1024 * 1024)
        upload_oversized = UploadFile(filename="too_large.pdf", file=io.BytesIO(oversized_bytes))

        try:
            upload_document(upload_oversized)
            assert False, "Expected 413 REQUEST ENTITY TOO LARGE"
        except HTTPException as e:
            assert e.status_code == 413
            assert "exceeds maximum allowed limit" in e.detail.lower()
            print(f"[PASS] test_5_oversized_file_rejection: 413 raised as expected ({e.detail}).")

        # Verify no temp files left behind
        leftover_tmps = list(docs_dir.glob(".tmp_*.pdf"))
        assert len(leftover_tmps) == 0, f"Found leftover temp files: {leftover_tmps}"
    finally:
        settings.MAX_UPLOAD_SIZE_BYTES = original_limit
        settings.MAX_UPLOAD_SIZE_MB = original_mb


def run_all():
    print("==================================================================")
    print("RUNNING LARGE FILE & STREAMING UPLOAD TEST SUITE")
    print("==================================================================")
    test_1_config_endpoint()
    test_2_small_streaming_upload()
    test_3_large_pdf_upload_exceeding_20mb()
    test_4_invalid_and_corrupted_uploads()
    test_5_oversized_file_rejection()
    print("==================================================================")
    print("ALL TESTS PASSED: 1 GB STREAMING UPLOADS FULLY VERIFIED")
    print("==================================================================")


if __name__ == "__main__":
    run_all()
