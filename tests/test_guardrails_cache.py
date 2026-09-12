import pytest
from langchain_core.documents import Document
from rag.guardrails.token_budget import TokenBudgetManager
from rag.cache.semantic_cache import SemanticCache


def test_token_budget_manager():
    budgeter = TokenBudgetManager(max_tokens=100)
    text = "Arthur Schopenhauer was a German philosopher."
    count = budgeter.count_tokens(text)
    assert count > 0

    docs = [
        Document(page_content="This is a small sentence.", metadata={"id": 1}),
        Document(page_content="This is another small sentence.", metadata={"id": 2}),
    ]
    fitted = budgeter.fit_documents_to_budget(docs, max_context_tokens=50)
    assert len(fitted) >= 1


def test_semantic_cache(tmp_path):
    db_path = tmp_path / "test_cache.db"
    cache = SemanticCache(db_path=db_path, similarity_threshold=0.85)

    q = "What is the wisdom of life?"
    payload = {"answer": "Wisdom of life is ordering our existence for happiness.", "citations": []}

    assert cache.get(q) is None
    cache.set(q, payload)

    cached_res = cache.get("What is the wisdom of life?")
    assert cached_res is not None
    assert cached_res["cached"] is True
    assert "Wisdom of life" in cached_res["answer"]
