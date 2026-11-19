from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Schema for an individual processed text chunk."""

    chunk_id: str = Field(description="Unique deterministic chunk identifier")
    document_id: str = Field(description="Parent document identifier")
    document_name: str = Field(description="Original filename of the parent document")
    page_number: int = Field(ge=1, description="1-indexed source page number in the original PDF")
    chunk_index: int = Field(ge=0, description="Sequential 0-indexed position within the document")
    chunk_text: str = Field(description="Normalized textual content of this chunk")
    token_count: int = Field(ge=0, description="Exact token count computed via BPE tokenizer")
    character_count: int = Field(ge=0, description="Character count of the chunk text")


class PageInfo(BaseModel):
    """Schema representing page-level extraction diagnostic info."""

    page_number: int = Field(ge=1, description="1-indexed page number")
    has_text: bool = Field(description="Whether extractable text was found on this page")
    character_count: int = Field(ge=0, description="Raw character count extracted from page")
    token_count: int = Field(ge=0, description="Token count extracted from page")


class DocumentInfo(BaseModel):
    """Schema representing ingested document catalog metadata."""

    document_id: str = Field(description="Unique document identifier (UUID)")
    document_name: str = Field(description="Sanitized original document filename")
    file_size_bytes: int = Field(ge=0, description="Size of the raw PDF file in bytes")
    content_hash: str = Field(description="SHA-256 hash of the PDF file content")
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of ingestion",
    )
    total_pages: int = Field(ge=0, description="Total number of pages in the PDF")
    pages_with_text: int = Field(ge=0, description="Count of pages containing extractable text")
    total_chunks: int = Field(ge=0, description="Number of text chunks generated")
    total_tokens: int = Field(ge=0, description="Total tokens across all chunks")
    storage_path: str = Field(description="Relative path to stored raw PDF")


class DocumentDetailResponse(DocumentInfo):
    """Detailed response including page diagnostics and all generated chunks."""

    pages: List[PageInfo] = Field(default_factory=list, description="Page extraction status")
    chunks: List[ChunkMetadata] = Field(default_factory=list, description="All generated chunks")


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful PDF ingestion."""

    message: str = Field(description="Status message")
    document: DocumentInfo = Field(description="Ingested document metadata")
    chunks_created: int = Field(ge=0, description="Total chunks created during ingestion")
    is_duplicate: bool = Field(default=False, description="Whether document already existed")


class DocumentDeleteResponse(BaseModel):
    """Response returned upon document deletion."""

    message: str = Field(description="Status message")
    document_id: str = Field(description="ID of deleted document")
