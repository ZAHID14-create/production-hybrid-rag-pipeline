from fastapi import APIRouter
from app.core.config import settings
from app.models.health import HealthResponse
from app.services.storage_service import storage_service

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns the operational status, application version, and component health of the Hybrid RAG API.",
)
def check_health() -> HealthResponse:
    """Return runtime diagnostic status."""
    gemini_status = "configured" if settings.GEMINI_API_KEY else "key_missing"
    doc_count = len(storage_service.list_documents())

    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        components={
            "api": "online",
            "fastapi_server": "active",
            "retrieval_pipeline": f"Phase 2A active (Documents: {doc_count})",
            "vector_store": "uninitialized (Phase 2B)",
            "gemini_api": gemini_status,
        },
    )
