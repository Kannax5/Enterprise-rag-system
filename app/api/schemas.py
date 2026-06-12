"""
schemas.py — Pydantic request/response models for all API endpoints.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ── /query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048, description="User's natural-language question.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")
    prompt_version: str = Field(default="v2", description="Prompt template version to use.")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the company's refund policy?",
                "top_k": 5,
                "prompt_version": "v2",
            }
        }


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    grounded: bool
    overlap_score: float


# ── /ingest ───────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket name.")
    prefix: str = Field(default="", description="S3 key prefix / folder path.")

    class Config:
        json_schema_extra = {
            "example": {
                "bucket": "my-enterprise-docs",
                "prefix": "knowledge-base/",
            }
        }


class IngestResponse(BaseModel):
    status: str
    message: Optional[str] = None
