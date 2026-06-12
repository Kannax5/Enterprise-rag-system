"""
test_retrieval.py — Unit tests for the retrieval layer.
"""

import numpy as np
import pytest

from app.embeddings.encoder import Encoder
from app.embeddings.faiss_store import FaissStore
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever


@pytest.fixture(scope="module")
def populated_store(tmp_path_factory):
    """Build and return a FaissStore loaded with 20 synthetic chunks."""
    tmp = tmp_path_factory.mktemp("faiss")
    encoder = Encoder()
    dim = encoder.embedding_dim
    store = FaissStore(
        index_path=str(tmp / "idx.faiss"),
        metadata_path=str(tmp / "meta.pkl"),
        embedding_dim=dim,
    )
    texts = [f"This is synthetic document number {i}." for i in range(20)]
    embeddings = encoder.encode(texts)
    metadata = [{"chunk_index": i, "text": texts[i], "source": "synthetic"} for i in range(20)]
    store.build(embeddings, metadata)
    store.save()
    return store, encoder


# ── Retriever ─────────────────────────────────────────────────────────────────

class TestRetriever:
    def test_retrieve_returns_top_k(self, populated_store):
        store, encoder = populated_store
        retriever = Retriever(encoder=encoder, faiss_store=store, top_k=5)
        results = retriever.retrieve("document", top_k=5)
        assert len(results) == 5

    def test_results_have_metadata_and_score(self, populated_store):
        store, encoder = populated_store
        retriever = Retriever(encoder=encoder, faiss_store=store)
        results = retriever.retrieve("synthetic number")
        for meta, score in results:
            assert "text" in meta
            assert isinstance(score, float)

    def test_scores_descending(self, populated_store):
        store, encoder = populated_store
        retriever = Retriever(encoder=encoder, faiss_store=store, top_k=10)
        results = retriever.retrieve("document")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ── Reranker ──────────────────────────────────────────────────────────────────

class TestReranker:
    def _make_results(self, scores):
        return [
            ({"text": f"chunk {i}", "source": "test"}, s)
            for i, s in enumerate(scores)
        ]

    def test_score_filter(self):
        reranker = Reranker(min_score=0.5)
        results = self._make_results([0.9, 0.6, 0.3, 0.1])
        filtered = reranker.rerank(results)
        assert all(s >= 0.5 for _, s in filtered)
        assert len(filtered) == 2

    def test_top_k_limit(self):
        reranker = Reranker(min_score=0.0)
        results = self._make_results([0.9, 0.8, 0.7, 0.6])
        filtered = reranker.rerank(results, top_k=2)
        assert len(filtered) == 2

    def test_mmr_returns_diverse_results(self):
        reranker = Reranker(mmr_lambda=0.5)
        dim = 8
        embeddings = [np.random.rand(dim).astype(np.float32) for _ in range(6)]
        results = self._make_results([0.9, 0.85, 0.8, 0.75, 0.7, 0.65])
        mmr_results = reranker.mmr_rerank(results, embeddings, top_k=3)
        assert len(mmr_results) == 3
