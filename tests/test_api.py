import pytest
from fastapi.testclient import TestClient
from rag.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["chroma_chunks"] > 0
    assert data["bm25_documents"] > 0


def test_query_endpoint(client):
    payload = {
        "question": "What does Schopenhauer say about happiness and Voltaire?",
        "top_k": 3,
        "enable_cache": True,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert "citations" in data
    assert len(data["citations"]) > 0
    assert data["latency_ms"] > 0
