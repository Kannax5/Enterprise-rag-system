"""
FastAPI application entry point for the Enterprise RAG System.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import query, ingest, health
from app.api.middleware import setup_middleware
from app.config import settings

app = FastAPI(
    title="Enterprise RAG System",
    description="Retrieval-Augmented Generation API powered by FAISS and a custom in-house LLM.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware (auth, logging, rate limiting)
setup_middleware(app)

# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    pass
