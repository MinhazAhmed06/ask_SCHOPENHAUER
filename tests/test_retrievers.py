import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from rag.vectorstore.bm25_store import BM25Store
from rag.retrievers.hybrid import HybridRetriever


def test_bm25_store(tmp_path):
    idx_path = tmp_path / "test_bm25.pkl"
    store = BM25Store(index_path=idx_path)
    
    docs = [
        Document(page_content="Schopenhauer writes about the doctrine of the will to live.", metadata={"id": 1}),
        Document(page_content="Voltaire discusses the nature of happiness and philosophy.", metadata={"id": 2}),
        Document(page_content="Kant developed transcendental idealism and the categorical imperative.", metadata={"id": 3}),
    ]
    
    store.fit(docs)
    assert store.count() == 3
    
    results = store.search("Voltaire happiness", k=2)
    assert len(results) > 0
    assert "Voltaire" in results[0][0].page_content
    assert results[0][1] > 0.0


def test_hybrid_rrf_scoring(tmp_path):
    bm25 = BM25Store(index_path=tmp_path / "bm25.pkl")
    docs = [
        Document(page_content="Schopenhauer on suicide and the will.", metadata={"chunk_id": "c1"}),
        Document(page_content="Aristotle on ethics and virtue.", metadata={"chunk_id": "c2"}),
    ]
    bm25.fit(docs)
    
    mock_dense = MagicMock()
    mock_dense.similarity_search_with_score.return_value = [
        (docs[0], 0.95),
        (docs[1], 0.40),
    ]

    hybrid = HybridRetriever(dense_store=mock_dense, sparse_store=bm25)
    results = hybrid.retrieve("suicide will", k=2)
    assert len(results) > 0
    assert results[0][0].metadata["chunk_id"] == "c1"
    assert results[0][1] > 0.0
