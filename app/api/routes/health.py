"""
health.py — GET /health: liveness probe.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """Returns 200 OK when the service is alive."""
    return HealthResponse(status="ok")
