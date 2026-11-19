from pathlib import Path
from typing import Dict, List, Tuple
import pymupdf

from app.core.config import settings
from app.models.document import PageInfo
from app.services.text_cleaner import clean_text


class PDFValidationError(Exception):
    """Custom exception raised when a PDF fails validation."""
    pass


class PDFPasswordProtectedError(PDFValidationError):
    """Raised when an uploaded PDF is password protected."""
    pass


class PDFCorruptedError(PDFValidationError):
    """Raised when an uploaded PDF is empty or corrupted."""
    pass


class PDFSizeExceededError(PDFValidationError):
    """Raised when a PDF exceeds the upload size limit."""
    pass


class PDFDocumentLoader:
    """Service for validating and extracting text from PDF documents using PyMuPDF."""

    @staticmethod
    def validate_pdf_bytes(file_bytes: bytes, filename: str) -> None:
        """
        Validates raw uploaded bytes before full parsing:
        - Checks filename extension
        - Validates size limit
        - Validates '%PDF-' magic byte header
        """
        # 1. Extension validation
        if not filename.lower().endswith(".pdf"):
            raise PDFValidationError(f"Invalid file extension for '{filename}'. Only .pdf files are accepted.")

        # 2. File size validation
        size = len(file_bytes)
        if size == 0:
            raise PDFCorruptedError(f"File '{filename}' is empty (0 bytes).")
        if size > settings.MAX_UPLOAD_SIZE_BYTES:
            max_mb = settings.MAX_UPLOAD_SIZE_MB
            actual_mb = size / (1024 * 1024)
            raise PDFSizeExceededError(
                f"File size ({actual_mb:.2f} MB) exceeds maximum allowed limit of {max_mb} MB."
            )

        # 3. Magic header check (standard PDF specification starts with %PDF-)
        if not file_bytes.startswith(b"%PDF-"):
            raise PDFCorruptedError(
                f"File '{filename}' is not a valid PDF document (missing %PDF- header)."
            )

    @classmethod
    def load_and_extract(cls, file_bytes: bytes, filename: str) -> Tuple[List[Dict[str, any]], List[PageInfo]]:
        """
        Extracts text from PDF bytes page-by-page preserving 1-indexed page numbers.
        Returns:
            Tuple of:
            - List of page text dicts: [{"page_number": 1, "text": "cleaned text..."}, ...]
            - List of PageInfo schemas for diagnostic reporting
        """
        cls.validate_pdf_bytes(file_bytes, filename)

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise PDFCorruptedError(f"Failed to parse PDF '{filename}': {str(e)}")

        with doc:
            # Check for encryption / password lock
            if doc.is_encrypted or doc.needs_pass:
                raise PDFPasswordProtectedError(
                    f"PDF '{filename}' is password-protected or encrypted. Please remove password protection before uploading."
                )

            total_pages = len(doc)
            if total_pages == 0:
                raise PDFCorruptedError(f"PDF '{filename}' contains no pages.")

            extracted_pages: List[Dict[str, any]] = []
            page_diagnostics: List[PageInfo] = []

            for page in doc:
                page_number = page.number + 1  # 1-indexed

                # Extract text using PyMuPDF's text layout engine
                raw_text = page.get_text("text") or ""
                cleaned = clean_text(raw_text)

                has_text = bool(cleaned.strip())
                char_count = len(cleaned)

                extracted_pages.append({
                    "page_number": page_number,
                    "text": cleaned,
                    "has_text": has_text,
                })

                page_diagnostics.append(
                    PageInfo(
                        page_number=page_number,
                        has_text=has_text,
                        character_count=char_count,
                        token_count=0,  # Will be calculated during tokenization
                    )
                )

            return extracted_pages, page_diagnostics

    @staticmethod
    def validate_pdf_file(file_path: Path, filename: str) -> None:
        """
        Validates a PDF file on disk before parsing without loading entire file into memory:
        - Checks filename extension
        - Validates file size from filesystem metadata
        - Validates '%PDF-' magic byte header by reading only the first 1024 bytes
        """
        if not filename.lower().endswith(".pdf"):
            raise PDFValidationError(f"Invalid file extension for '{filename}'. Only .pdf files are accepted.")

        if not file_path.exists():
            raise PDFCorruptedError(f"Uploaded file '{filename}' was not found on disk.")

        size = file_path.stat().st_size
        if size == 0:
            raise PDFCorruptedError(f"File '{filename}' is empty (0 bytes).")

        if size > settings.MAX_UPLOAD_SIZE_BYTES:
            max_mb = settings.MAX_UPLOAD_SIZE_MB
            actual_mb = size / (1024 * 1024)
            raise PDFSizeExceededError(
                f"File size ({actual_mb:.2f} MB) exceeds maximum allowed limit of {max_mb} MB."
            )

        with open(file_path, "rb") as f:
            header = f.read(1024)
            if not header.startswith(b"%PDF-"):
                raise PDFCorruptedError(
                    f"File '{filename}' is not a valid PDF document (missing %PDF- header)."
                )

    @classmethod
    def load_and_extract_from_file(cls, file_path: Path, filename: str) -> Tuple[List[Dict[str, any]], List[PageInfo]]:
        """
        Extracts text from a PDF file path page-by-page using PyMuPDF memory mapping.
        Does not buffer the entire document in Python memory, making it safe for 1 GB+ files.
        """
        cls.validate_pdf_file(file_path, filename)

        try:
            doc = pymupdf.open(str(file_path))
        except Exception as e:
            raise PDFCorruptedError(f"Failed to parse PDF '{filename}': {str(e)}")

        with doc:
            # Check for encryption / password lock
            if doc.is_encrypted or doc.needs_pass:
                raise PDFPasswordProtectedError(
                    f"PDF '{filename}' is password-protected or encrypted. Please remove password protection before uploading."
                )

            total_pages = len(doc)
            if total_pages == 0:
                raise PDFCorruptedError(f"PDF '{filename}' contains no pages.")

            extracted_pages: List[Dict[str, any]] = []
            page_diagnostics: List[PageInfo] = []

            for page in doc:
                page_number = page.number + 1  # 1-indexed

                # Extract text using PyMuPDF's text layout engine
                raw_text = page.get_text("text") or ""
                cleaned = clean_text(raw_text)

                has_text = bool(cleaned.strip())
                char_count = len(cleaned)

                extracted_pages.append({
                    "page_number": page_number,
                    "text": cleaned,
                    "has_text": has_text,
                })

                page_diagnostics.append(
                    PageInfo(
                        page_number=page_number,
                        has_text=has_text,
                        character_count=char_count,
                        token_count=0,
                    )
                )

            return extracted_pages, page_diagnostics
