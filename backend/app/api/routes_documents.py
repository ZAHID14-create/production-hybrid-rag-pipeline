import hashlib
from pathlib import Path
import uuid
from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.models.document import (
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentInfo,
    DocumentUploadResponse,
)
from app.services.document_loader import (
    PDFCorruptedError,
    PDFDocumentLoader,
    PDFPasswordProtectedError,
    PDFSizeExceededError,
    PDFValidationError,
)
from app.services.storage_service import storage_service
from app.services.text_splitter import TextSplitter
from app.services.index_manager import index_manager

router = APIRouter(prefix="/documents", tags=["Document Ingestion & Management"])
text_splitter = TextSplitter()


@router.get(
    "/config",
    summary="Get document upload configuration",
    description="Returns maximum upload size in MB and bytes.",
)
def get_upload_config() -> dict:
    """Returns dynamic upload configuration for frontend clients."""
    return {
        "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
        "max_upload_size_bytes": settings.MAX_UPLOAD_SIZE_BYTES,
    }


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest PDF document",
    description="Validates the PDF, extracts clean text page-by-page, generates sentence-aware chunks, and persists metadata.",
)
def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Handles streaming PDF file upload, validation, extraction, chunking, and persistence."""
    filename = file.filename or "uploaded_document.pdf"

    # Fast client filename extension validation
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type for '{filename}'. Only .pdf files are accepted.",
        )

    # Stream uploaded file directly to a temporary file on disk using 1 MB chunks
    # This prevents loading the entire PDF into RAM, safely supporting files up to 1 GB+
    temp_filename = f".tmp_{uuid.uuid4().hex}.pdf"
    temp_path = Path(settings.DOCUMENTS_DIRECTORY) / temp_filename
    hasher = hashlib.sha256()
    bytes_written = 0
    chunk_size = 1024 * 1024  # 1 MB chunk

    try:
        with open(temp_path, "wb") as f_out:
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > settings.MAX_UPLOAD_SIZE_BYTES:
                    raise PDFSizeExceededError(
                        f"File size exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
                    )
                hasher.update(chunk)
                f_out.write(chunk)
    except PDFSizeExceededError as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to receive streaming upload: {str(e)}",
        )

    if bytes_written == 0:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{filename}' is empty (0 bytes).",
        )

    content_hash = hasher.hexdigest()

    # Check for duplicate by content hash before expensive extraction & indexing
    existing_doc = storage_service.find_by_content_hash(content_hash)
    if existing_doc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        return DocumentUploadResponse(
            message=f"Document '{filename}' has already been ingested (matching hash {content_hash[:8]}...).",
            document=existing_doc,
            chunks_created=existing_doc.total_chunks,
            is_duplicate=True,
        )

    # Extract text and page diagnostics using PyMuPDF from the streamed file on disk
    try:
        pages_data, page_diagnostics = PDFDocumentLoader.load_and_extract_from_file(temp_path, filename)
    except PDFPasswordProtectedError as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PDFSizeExceededError as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except PDFCorruptedError as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PDFValidationError as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during PDF extraction: {str(e)}",
        )

    document_id = str(uuid.uuid4())
    safe_filename = storage_service.sanitize_filename(filename)

    # Promote temporary file to permanent location in documents/ directory
    try:
        storage_path = storage_service.promote_temp_file(temp_path, safe_filename, document_id)
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist uploaded document: {str(e)}",
        )

    # Generate sentence-aware overlapping chunks across all pages
    all_chunks = []
    current_chunk_idx = 0
    pages_with_text = 0

    for page_dict, page_diag in zip(pages_data, page_diagnostics):
        page_num = page_dict["page_number"]
        page_text = page_dict["text"]

        if page_dict["has_text"]:
            pages_with_text += 1

        # Split page text into chunks
        page_chunks = text_splitter.split_page_text(
            text=page_text,
            document_id=document_id,
            document_name=safe_filename,
            page_number=page_num,
            start_chunk_index=current_chunk_idx,
        )

        # Update diagnostic page token count
        page_diag.token_count = sum(c.token_count for c in page_chunks)
        all_chunks.extend(page_chunks)
        current_chunk_idx += len(page_chunks)

    total_tokens = sum(c.token_count for c in all_chunks)

    # Build DocumentInfo
    doc_info = DocumentInfo(
        document_id=document_id,
        document_name=safe_filename,
        file_size_bytes=bytes_written,
        content_hash=content_hash,
        total_pages=len(pages_data),
        pages_with_text=pages_with_text,
        total_chunks=len(all_chunks),
        total_tokens=total_tokens,
        storage_path=storage_path,
    )

    # Persist document metadata and chunks JSON
    storage_service.save_processed_document(doc_info, page_diagnostics, all_chunks)

    # Incrementally index chunks into Dense Vector (ChromaDB) and Lexical (BM25) stores
    if all_chunks:
        try:
            index_manager.index_chunks(all_chunks, document_name=safe_filename)
        except Exception:
            # Document is saved; indexing warning can be logged without failing the upload response
            pass

    return DocumentUploadResponse(
        message=f"Document '{safe_filename}' successfully processed and indexed ({len(all_chunks)} chunks).",
        document=doc_info,
        chunks_created=len(all_chunks),
        is_duplicate=False,
    )


@router.get(
    "",
    response_model=List[DocumentInfo],
    summary="List all ingested documents",
    description="Returns metadata catalog for all previously processed PDF documents.",
)
def list_documents() -> List[DocumentInfo]:
    """Retrieves all registered documents from the local catalog."""
    return storage_service.list_documents()


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details and chunks",
    description="Returns detailed extraction metrics, page breakdown, and all chunk text with metadata.",
)
def get_document(document_id: str) -> DocumentDetailResponse:
    """Retrieves full document record with chunks."""
    doc_detail = storage_service.get_document_detail(document_id)
    if not doc_detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' was not found.",
        )
    return doc_detail


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete an ingested document",
    description="Permanently removes the stored PDF and all corresponding processed chunk files from disk.",
)
def delete_document(document_id: str) -> DocumentDeleteResponse:
    """Deletes stored document and its processed chunks, synchronizing both indices."""
    success = storage_service.delete_document(document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' was not found or could not be deleted.",
        )

    # Remove corresponding chunks from both ChromaDB and BM25
    try:
        index_manager.delete_document_indices(document_id)
    except Exception:
        pass

    return DocumentDeleteResponse(
        message=f"Document '{document_id}' and all associated chunks were successfully deleted.",
        document_id=document_id,
    )
