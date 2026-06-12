"""
pdf_parser.py — PyMuPDF-based PDF text extractor.
"""

import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFParser:
    """Extracts plain text from PDF files page by page."""

    def __init__(self, extract_images: bool = False):
        self.extract_images = extract_images

    def parse(self, file_path: str | Path) -> List[dict]:
        """
        Parse a PDF and return a list of page dicts:
            {
                "page_number": int,
                "text": str,
                "source": str,   # absolute file path
            }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        pages: List[dict] = []
        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            logger.error("Cannot open PDF %s: %s", file_path, exc)
            raise

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if not text:
                logger.debug("Page %d of %s is empty — skipping.", page_num + 1, file_path.name)
                continue
            pages.append(
                {
                    "page_number": page_num + 1,
                    "text": text,
                    "source": str(file_path.resolve()),
                }
            )

        doc.close()
        logger.info("Extracted %d page(s) from %s", len(pages), file_path.name)
        return pages
