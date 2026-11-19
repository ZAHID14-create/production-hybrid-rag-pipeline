import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.models.document import ChunkMetadata
from app.models.index import (
    IndexRebuildResponse,
    IndexStatsResponse,
    IndexStatusResponse,
)
from app.services.bm25_service import bm25_service
from app.services.storage_service import storage_service
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)


class IndexManager:
    """Orchestration service synchronizing Dense Vector (ChromaDB) and Lexical (BM25) indices.

    Coordinates incremental indexing on upload, synchronized deletion on document removal,
    and complete rebuild from persistent processed JSON files.
    """

    def __init__(self):
        self._status: str = "idle"  # idle, indexing, rebuilding, error
        self._last_action: str = "System initialized"
        self._last_indexed_at: Optional[datetime] = None
        self._last_error: Optional[str] = None

    def get_status(self) -> IndexStatusResponse:
        """Returns the current operational status of the index orchestrator."""
        docs = storage_service.list_documents()
        total_chunks = sum(d.total_chunks for d in docs)
        return IndexStatusResponse(
            status=self._status,
            last_action=self._last_action,
            last_indexed_at=self._last_indexed_at,
            last_error=self._last_error,
            total_documents=len(docs),
            total_chunks=total_chunks,
        )

    def get_stats(self) -> IndexStatsResponse:
        """Returns detailed synchronized statistics for both ChromaDB and BM25."""
        chroma_info = vector_store.get_stats()
        bm25_info = bm25_service.get_stats()

        chroma_count = chroma_info.get("chunk_count", 0)
        bm25_count = bm25_info.get("chunk_count", 0)

        # In a fully healthy pipeline, both indices have identical chunk counts
        is_synchronized = (chroma_count == bm25_count)
        total_indexed = max(chroma_count, bm25_count)

        return IndexStatsResponse(
            chroma_status=chroma_info.get("status", "empty"),
            chroma_collection_name=chroma_info.get("collection_name", "rag_chunks"),
            chroma_chunk_count=chroma_count,
            embedding_model_name=chroma_info.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_dimension=chroma_info.get("embedding_dimension", 384),
            bm25_status=bm25_info.get("status", "empty"),
            bm25_chunk_count=bm25_count,
            bm25_corpus_size_bytes=bm25_info.get("corpus_size_bytes", 0),
            total_indexed_chunks=total_indexed,
            is_synchronized=is_synchronized,
        )

    def index_chunks(self, chunks: List[ChunkMetadata], document_name: str = "") -> int:
        """Indexes an explicit list of chunks into both ChromaDB and BM25."""
        if not chunks:
            return 0

        self._status = "indexing"
        self._last_action = f"Indexing {len(chunks)} chunks from '{document_name}'"
        try:
            # 1. Index in ChromaDB (dense vectors)
            vector_store.add_chunks(chunks)

            # 2. Index in BM25 (lexical tokens)
            bm25_service.add_chunks(chunks)

            self._status = "idle"
            self._last_indexed_at = datetime.now(timezone.utc)
            self._last_error = None
            self._last_action = f"Successfully indexed {len(chunks)} chunks from '{document_name}'"
            return len(chunks)
        except Exception as e:
            self._status = "error"
            self._last_error = str(e)
            self._last_action = f"Failed indexing chunks from '{document_name}': {str(e)}"
            logger.error("Indexing failed for '%s': %s", document_name, e)
            raise

    def index_document(self, document_id: str) -> int:
        """Reads document chunks from storage and adds them to both indices."""
        doc_detail = storage_service.get_document_detail(document_id)
        if not doc_detail:
            raise ValueError(f"Document with ID '{document_id}' not found in storage.")

        if not doc_detail.chunks:
            logger.info("Document '%s' has 0 chunks to index.", doc_detail.document_name)
            return 0

        return self.index_chunks(doc_detail.chunks, document_name=doc_detail.document_name)

    def delete_document_indices(self, document_id: str) -> Dict[str, int]:
        """Removes all document chunks from both ChromaDB and BM25."""
        chroma_deleted = vector_store.delete_document(document_id)
        bm25_deleted = bm25_service.delete_document(document_id)

        self._last_action = f"Deleted document '{document_id}' (Chroma: {chroma_deleted}, BM25: {bm25_deleted})"
        return {"chroma_deleted": chroma_deleted, "bm25_deleted": bm25_deleted}

    def rebuild_all_indices(self) -> IndexRebuildResponse:
        """Rebuilds both ChromaDB and BM25 indices from processed documents."""
        start_time = time.time()
        self._status = "rebuilding"
        self._last_action = "Rebuilding all indices from processed documents catalog"

        try:
            # 1. Reset both indices
            vector_store.reset_collection()
            bm25_service.reset_index()

            # 2. Gather all documents from storage catalog
            documents = storage_service.list_documents()
            total_chunks_indexed = 0

            for doc in documents:
                doc_detail = storage_service.get_document_detail(doc.document_id)
                if doc_detail and doc_detail.chunks:
                    # Index in both
                    vector_store.add_chunks(doc_detail.chunks)
                    bm25_service.add_chunks(doc_detail.chunks)
                    total_chunks_indexed += len(doc_detail.chunks)

            elapsed = round(time.time() - start_time, 2)
            self._status = "idle"
            self._last_indexed_at = datetime.now(timezone.utc)
            self._last_error = None
            self._last_action = (
                f"Rebuild completed: {len(documents)} documents, "
                f"{total_chunks_indexed} chunks in {elapsed}s"
            )

            return IndexRebuildResponse(
                message=f"Successfully rebuilt all indices: {len(documents)} documents and {total_chunks_indexed} chunks indexed.",
                documents_processed=len(documents),
                chunks_indexed=total_chunks_indexed,
                duration_seconds=elapsed,
                status="completed",
            )
        except Exception as e:
            self._status = "error"
            self._last_error = str(e)
            self._last_action = f"Rebuild failed: {str(e)}"
            logger.error("Index rebuild error: %s", e)
            raise


index_manager = IndexManager()
