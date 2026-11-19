import logging
from fastapi import APIRouter, HTTPException, status

from app.models.retrieval import RetrievalRequest, RetrievalResponse
from app.services.retrieval_service import retrieval_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieval", tags=["Hybrid Retrieval & Fusion"])


@router.post(
    "/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Search chunks using BM25, Dense, or Hybrid RRF",
    description="Performs keyword BM25 retrieval, semantic dense vector retrieval, or Reciprocal Rank Fusion hybrid retrieval.",
)
def search_documents(request: RetrievalRequest) -> RetrievalResponse:
    """Executes search query across knowledge base with requested retrieval strategy."""
    try:
        return retrieval_service.search(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Retrieval failed for query '%s': %s", request.query, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval search failed: {str(e)}",
        )
