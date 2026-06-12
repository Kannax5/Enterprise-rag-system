"""
test_ingestion.py — Unit tests for the ingestion pipeline.
"""

import os
import tempfile
from pathlib import Path

import pytest

from app.ingestion.chunker import Chunker
from app.ingestion.csv_parser import CSVParser
from app.ingestion.metadata_tagger import MetadataTagger
from app.ingestion.pdf_parser import PDFParser


# ── Chunker ───────────────────────────────────────────────────────────────────

class TestChunker:
    def test_basic_chunking(self):
        chunker = Chunker(chunk_size=10, chunk_overlap=2)
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "text" in chunk
            assert "chunk_index" in chunk

    def test_empty_text_returns_empty(self):
        chunker = Chunker()
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_metadata_forwarded(self):
        chunker = Chunker()
        meta = {"source": "test.pdf", "page_number": 1}
        chunks = chunker.chunk_text("Hello world. Goodbye world.", metadata=meta)
        for chunk in chunks:
            assert chunk["source"] == "test.pdf"
            assert chunk["page_number"] == 1


# ── CSVParser ─────────────────────────────────────────────────────────────────

class TestCSVParser:
    def test_parse_basic_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")
        parser = CSVParser()
        records = parser.parse(csv_file)
        assert len(records) == 2
        assert "Alice" in records[0]["text"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            CSVParser().parse("/nonexistent/file.csv")

    def test_custom_columns(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age,city\nAlice,30,NYC\n", encoding="utf-8")
        parser = CSVParser(text_columns=["name", "city"])
        records = parser.parse(csv_file)
        assert "age" not in records[0]["text"]
        assert "NYC" in records[0]["text"]


# ── MetadataTagger ────────────────────────────────────────────────────────────

class TestMetadataTagger:
    def test_tags_applied(self):
        tagger = MetadataTagger()
        chunks = [{"chunk_index": 0, "text": "hello", "source": "test.pdf"}]
        tagged = tagger.tag(chunks)
        assert tagged[0]["doc_type"] == "pdf"
        assert "ingested_at" in tagged[0]
        assert "chunk_id" in tagged[0]

    def test_chunk_id_deterministic(self):
        tagger = MetadataTagger()
        chunk = {"chunk_index": 0, "text": "hello", "source": "test.pdf"}
        id1 = tagger.tag([dict(chunk)])[0]["chunk_id"]
        id2 = tagger.tag([dict(chunk)])[0]["chunk_id"]
        assert id1 == id2


# ── PDFParser (integration — requires a real PDF) ─────────────────────────────

class TestPDFParser:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            PDFParser().parse("/nonexistent/file.pdf")
