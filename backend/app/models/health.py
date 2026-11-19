from datetime import datetime, timezone
from typing import Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for system health check response."""

    status: str = Field(default="healthy", description="Overall service status")
    version: str = Field(description="Application version")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the check",
    )
    environment: str = Field(description="Runtime environment (development, production)")
    components: Dict[str, Any] = Field(
        default_factory=lambda: {
            "api": "online",
            "retrieval_pipeline": "standby (Phase 1)",
            "vector_store": "uninitialized (Phase 2)",
            "gemini_api": "pending_configuration",
        },
        description="Health status of downstream sub-systems",
    )
