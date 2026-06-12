"""
test_embeddings.py — Unit tests for the embeddings layer.
"""

import numpy as np
import pytest

from app.embeddings.encoder import Encoder
from app.embeddings.faiss_store import FaissStore
from app.embeddings.index_builder import IndexBuilder


# ── Encoder ───────────────────────────────────────────────────────────────────

class TestEncoder:
    @pytest.fixture(scope="class")
    def encoder(self):
        return Encoder()

    def test_encode_returns_ndarray(self, encoder):
        vecs = encoder.encode(["Hello, world!"])
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (1, encoder.embedding_dim)

    def test_encode_empty_returns_empty(self, encoder):
        vecs = encoder.encode([])
        assert vecs.shape[0] == 0

    def test_encode_query_returns_1d(self, encoder):
        vec = encoder.encode_query("test query")
        assert vec.ndim == 1
        assert len(vec) == encoder.embedding_dim

    def test_vectors_are_normalized(self, encoder):
        vecs = encoder.encode(["normalised vector test"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ── FaissStore ────────────────────────────────────────────────────────────────

class TestFaissStore:
    @pytest.fixture
    def store_with_data(self, tmp_path):
        dim = 8
        store = FaissStore(
            index_path=str(tmp_path / "idx.faiss"),
            metadata_path=str(tmp_path / "meta.pkl"),
            embedding_dim=dim,
        )
        embeddings = np.random.rand(10, dim).astype(np.float32)
        metadata = [{"chunk_index": i, "text": f"chunk {i}"} for i in range(10)]
        store.build(embeddings, metadata)
        return store, dim

    def test_build_sets_total_vectors(self, store_with_data):
        store, _ = store_with_data
        assert store.total_vectors == 10

    def test_query_returns_results(self, store_with_data):
        store, dim = store_with_data
        q = np.random.rand(dim).astype(np.float32)
        results = store.query(q, top_k=3)
        assert len(results) == 3
        for meta, score in results:
            assert "text" in meta
            assert isinstance(score, float)

    def test_save_and_load(self, store_with_data, tmp_path):
        store, dim = store_with_data
        store.save()

        new_store = FaissStore(
            index_path=str(store.index_path),
            metadata_path=str(store.metadata_path),
            embedding_dim=dim,
        )
        new_store.load()
        assert new_store.total_vectors == 10


# ── IndexBuilder ──────────────────────────────────────────────────────────────

class TestIndexBuilder:
    def test_build_from_chunks(self, tmp_path):
        encoder = Encoder()
        store = FaissStore(
            index_path=str(tmp_path / "idx.faiss"),
            metadata_path=str(tmp_path / "meta.pkl"),
            embedding_dim=encoder.embedding_dim,
        )
        builder = IndexBuilder(encoder=encoder, faiss_store=store)
        chunks = [{"text": f"document chunk number {i}", "source": "test"} for i in range(5)]
        result_store = builder.build(chunks, save=True)
        assert result_store.total_vectors == 5
