import logging
from fastapi import APIRouter, HTTPException, status

from app.models.index import (
    IndexRebuildResponse,
    IndexStatsResponse,
    IndexStatusResponse,
)
from app.services.index_manager import index_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/index", tags=["Index Management"])


@router.get(
    "/status",
    response_model=IndexStatusResponse,
    summary="Get indexing status",
    description="Returns current orchestrator operational status, last action, and catalog totals.",
)
def get_index_status() -> IndexStatusResponse:
    """Retrieves operational status of the indexing orchestrator."""
    return index_manager.get_status()


@router.get(
    "/stats",
    response_model=IndexStatsResponse,
    summary="Get indexing statistics",
    description="Returns comprehensive stats for both Dense Vector (ChromaDB) and Lexical (BM25) indices.",
)
def get_index_stats() -> IndexStatsResponse:
    """Retrieves detailed stats for ChromaDB collection and BM25 index."""
    return index_manager.get_stats()


@router.post(
    "/rebuild",
    response_model=IndexRebuildResponse,
    summary="Rebuild all indices",
    description="Clears and rebuilds both ChromaDB and BM25 indices from stored processed document chunks.",
)
def rebuild_indices() -> IndexRebuildResponse:
    """Reconstructs all indices from persistent document storage."""
    try:
        return index_manager.rebuild_all_indices()
    except Exception as e:
        logger.error("Failed to rebuild indices: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index rebuild failed: {str(e)}",
        )
