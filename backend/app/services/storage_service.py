import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import uuid

from app.core.config import settings
from app.models.document import DocumentInfo, DocumentDetailResponse, ChunkMetadata, PageInfo


class StorageService:
    """Manages file persistence for raw PDFs and processed document chunk data."""

    def __init__(self):
        self.docs_dir = Path(settings.DOCUMENTS_DIRECTORY)
        self.processed_dir = Path(settings.PROCESSED_DATA_DIRECTORY)
        self.catalog_file = self.processed_dir / "catalog.json"

        # Ensure target directories exist
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        if not self.catalog_file.exists():
            self._save_catalog({})

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitizes filenames to prevent directory traversal and remove problematic characters.
        """
        # Strip path components
        base = os.path.basename(filename)
        # Remove any path traversal or invalid characters
        clean = re.sub(r"[^\w\-. ]", "_", base).strip()
        # Fallback if empty
        return clean or "document.pdf"

    @staticmethod
    def compute_sha256(data: bytes) -> str:
        """Computes SHA-256 hexadecimal digest of byte data."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def compute_sha256_of_file(file_path: Path, chunk_size: int = 65536) -> str:
        """Computes SHA-256 hexadecimal digest of a file incrementally in 64 KB blocks."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_catalog(self) -> Dict[str, dict]:
        """Loads the document catalog from disk."""
        if not self.catalog_file.exists():
            return {}
        try:
            with open(self.catalog_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_catalog(self, catalog: Dict[str, dict]) -> None:
        """Atomically saves the document catalog to disk."""
        temp_file = self.catalog_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, default=str)
        temp_file.replace(self.catalog_file)

    def find_by_content_hash(self, content_hash: str) -> Optional[DocumentInfo]:
        """Checks if a document with the identical content hash already exists."""
        catalog = self._load_catalog()
        for doc_dict in catalog.values():
            if doc_dict.get("content_hash") == content_hash:
                return DocumentInfo(**doc_dict)
        return None

    def save_raw_pdf(self, file_bytes: bytes, filename: str, doc_id: str) -> str:
        """
        Saves raw PDF bytes into the documents/ directory using safe filename.
        Returns the relative path from project root.
        """
        safe_name = self.sanitize_filename(filename)
        stored_name = f"{doc_id}_{safe_name}"
        dest_path = self.docs_dir / stored_name

        with open(dest_path, "wb") as f:
            f.write(file_bytes)

        return str(dest_path.relative_to(self.docs_dir.parent))

    def promote_temp_file(self, temp_path: Path, filename: str, doc_id: str) -> str:
        """
        Promotes an incrementally uploaded temporary file into the documents/ directory
        under its permanent {doc_id}_{safe_filename} path.
        Returns the relative path from project root.
        """
        safe_name = self.sanitize_filename(filename)
        stored_name = f"{doc_id}_{safe_name}"
        dest_path = self.docs_dir / stored_name

        shutil.move(str(temp_path), str(dest_path))
        return str(dest_path.relative_to(self.docs_dir.parent))

    def save_processed_document(
        self,
        doc_info: DocumentInfo,
        pages: List[PageInfo],
        chunks: List[ChunkMetadata],
    ) -> None:
        """
        Persists detailed document data (metadata, page diagnostics, and chunks) as JSON
        and registers the document in the central catalog.
        """
        doc_detail = DocumentDetailResponse(
            **doc_info.model_dump(),
            pages=pages,
            chunks=chunks,
        )

        # 1. Save detailed document JSON
        detail_path = self.processed_dir / f"{doc_info.document_id}.json"
        temp_path = detail_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(doc_detail.model_dump_json(indent=2))
        temp_path.replace(detail_path)

        # 2. Update catalog manifest
        catalog = self._load_catalog()
        catalog[doc_info.document_id] = doc_info.model_dump(mode="json")
        self._save_catalog(catalog)

    def list_documents(self) -> List[DocumentInfo]:
        """Returns metadata for all ingested documents in the catalog."""
        catalog = self._load_catalog()
        docs: List[DocumentInfo] = []
        for doc_data in catalog.values():
            try:
                docs.append(DocumentInfo(**doc_data))
            except Exception:
                continue
        # Sort descending by upload timestamp
        docs.sort(key=lambda d: d.uploaded_at, reverse=True)
        return docs

    def get_document_detail(self, doc_id: str) -> Optional[DocumentDetailResponse]:
        """Loads full document detail including page diagnostics and all chunks."""
        detail_path = self.processed_dir / f"{doc_id}.json"
        if not detail_path.exists():
            return None

        try:
            with open(detail_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DocumentDetailResponse(**data)
        except Exception:
            return None

    def delete_document(self, doc_id: str) -> bool:
        """
        Safely deletes the stored PDF, the processed chunks JSON, and catalog entry.
        Returns True if document existed and was deleted, False otherwise.
        """
        catalog = self._load_catalog()
        doc_dict = catalog.get(doc_id)
        if not doc_dict and not (self.processed_dir / f"{doc_id}.json").exists():
            return False

        # 1. Delete raw PDF if it exists
        if doc_dict and "storage_path" in doc_dict:
            raw_path = self.docs_dir.parent / doc_dict["storage_path"]
            if raw_path.exists():
                try:
                    raw_path.unlink()
                except Exception:
                    pass

        # 2. Delete processed JSON
        detail_path = self.processed_dir / f"{doc_id}.json"
        if detail_path.exists():
            try:
                detail_path.unlink()
            except Exception:
                pass

        # 3. Remove from catalog
        if doc_id in catalog:
            del catalog[doc_id]
            self._save_catalog(catalog)

        return True


storage_service = StorageService()
