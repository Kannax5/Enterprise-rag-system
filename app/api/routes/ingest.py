"""
ingest.py — POST /api/v1/ingest: trigger document ingestion pipeline.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.api.schemas import IngestRequest, IngestResponse
from app.embeddings.encoder import Encoder
from app.embeddings.faiss_store import FaissStore
from app.embeddings.index_builder import IndexBuilder
from app.ingestion.chunker import Chunker
from app.ingestion.csv_parser import CSVParser
from app.ingestion.metadata_tagger import MetadataTagger
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.s3_loader import S3Loader

logger = logging.getLogger(__name__)

router = APIRouter()


def _run_ingestion(bucket: str, prefix: str) -> None:
    """Background task: load → parse → chunk → tag → embed → index."""
    loader = S3Loader(bucket=bucket, prefix=prefix)
    chunker = Chunker()
    tagger = MetadataTagger()
    encoder = Encoder()
    store = FaissStore(embedding_dim=encoder.embedding_dim)
    builder = IndexBuilder(encoder=encoder, faiss_store=store)

    all_chunks = []

    for local_path in loader.download_all():
        ext = local_path.suffix.lower()
        try:
            if ext == ".pdf":
                pages = PDFParser().parse(local_path)
                for page in pages:
                    all_chunks.extend(
                        tagger.tag(chunker.chunk_text(page["text"], metadata=page))
                    )
            elif ext == ".csv":
                rows = CSVParser().parse(local_path)
                for row in rows:
                    all_chunks.extend(
                        tagger.tag(chunker.chunk_text(row["text"], metadata=row))
                    )
            else:
                text = Path(local_path).read_text(encoding="utf-8", errors="ignore")
                meta = {"source": str(local_path)}
                all_chunks.extend(tagger.tag(chunker.chunk_text(text, metadata=meta)))
        except Exception as exc:
            logger.error("Failed to process %s: %s", local_path, exc)

    if all_chunks:
        builder.build(all_chunks, save=True)
        logger.info("Ingestion complete: %d chunks indexed.", len(all_chunks))
    else:
        logger.warning("No chunks produced — index not updated.")


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger S3 document ingestion and index build",
)
async def ingest_endpoint(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    try:
        background_tasks.add_task(_run_ingestion, request.bucket, request.prefix)
        return IngestResponse(
            status="accepted",
            message=f"Ingestion started for s3://{request.bucket}/{request.prefix}",
        )
    except Exception as exc:
        logger.exception("Failed to schedule ingestion: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start ingestion.",
        )
