"""
build_index.py — One-time script: ingest all S3 docs and build the FAISS index.

Usage:
    python scripts/build_index.py [--bucket BUCKET] [--prefix PREFIX]
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.embeddings.encoder import Encoder
from app.embeddings.faiss_store import FaissStore
from app.embeddings.index_builder import IndexBuilder
from app.ingestion.chunker import Chunker
from app.ingestion.csv_parser import CSVParser
from app.ingestion.metadata_tagger import MetadataTagger
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.s3_loader import S3Loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from S3 documents.")
    parser.add_argument("--bucket", default=settings.S3_BUCKET_NAME, help="S3 bucket name")
    parser.add_argument("--prefix", default=settings.S3_PREFIX, help="S3 key prefix")
    parser.add_argument("--local", action="store_true", help="Build index from local data/raw directory directly, bypassing S3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    encoder = Encoder()
    store = FaissStore(embedding_dim=encoder.embedding_dim)
    builder = IndexBuilder(encoder=encoder, faiss_store=store)

    chunker = Chunker()
    tagger = MetadataTagger()

    # Determine if we run in local mode or S3 mode
    is_local = (
        args.local
        or settings.AWS_ACCESS_KEY_ID in ("", "your_aws_access_key_here")
        or settings.S3_BUCKET_NAME in ("", "your-enterprise-docs-bucket")
    )

    if is_local:
        logger.info("Local mode activated or AWS credentials not set. Scanning data/raw/...")
        local_dir = Path("data/raw")
        local_files = [
            p for p in local_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".pdf", ".csv", ".txt"}
        ]
    else:
        logger.info("S3 mode activated. Downloading documents from s3://%s/%s", args.bucket, args.prefix)
        loader = S3Loader(bucket=args.bucket, prefix=args.prefix)
        try:
            local_files = list(loader.download_all())
        except Exception as exc:
            logger.error("Failed to download files from S3: %s. Falling back to local mode.", exc)
            local_dir = Path("data/raw")
            local_files = [
                p for p in local_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".pdf", ".csv", ".txt"}
            ]

    all_chunks = []

    for local_path in local_files:
        ext = local_path.suffix.lower()
        logger.info("Processing: %s", local_path.name)
        file_chunks = []
        try:
            if ext == ".pdf":
                pages = PDFParser().parse(local_path)
                for page in pages:
                    file_chunks.extend(
                        tagger.tag(chunker.chunk_text(page["text"], metadata=page))
                    )
            elif ext == ".csv":
                rows = CSVParser().parse(local_path)
                for row in rows:
                    file_chunks.extend(
                        tagger.tag(chunker.chunk_text(row["text"], metadata=row))
                    )
            else:
                text = local_path.read_text(encoding="utf-8", errors="ignore")
                meta = {"source": str(local_path)}
                file_chunks.extend(tagger.tag(chunker.chunk_text(text, metadata=meta)))
            
            logger.info("Total chunks generated for file %s: %d", local_path.name, len(file_chunks))
            for chunk in file_chunks:
                chunk_id = chunk.get("chunk_id", "unknown")
                source_path = chunk.get("source", "unknown")
                snippet = chunk.get("text", "")[:200].replace('\n', ' ')
                logger.info("  [Chunk ID: %s] [Source: %s] Snippet: %s...", chunk_id, source_path, snippet)
                
            all_chunks.extend(file_chunks)
        except Exception as exc:
            logger.error("Skipping %s due to error: %s", local_path.name, exc)

    if not all_chunks:
        logger.error("No chunks produced. Aborting.")
        sys.exit(1)

    logger.info("Total chunks: %d. Building index…", len(all_chunks))
    builder.build(all_chunks, save=True)
    logger.info("Done. Index saved to %s", settings.FAISS_INDEX_PATH)


if __name__ == "__main__":
    main()
