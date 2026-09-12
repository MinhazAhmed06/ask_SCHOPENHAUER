import json
import sqlite3
import time
from pathlib import Path
import numpy as np
from ..config import settings
from ..embeddings.factory import get_embedding_model


class SemanticCache:
    """
    Semantic response cache backed by SQLite and dense vector cosine similarity.
    Provides sub-millisecond retrieval and $0 token cost for semantically identical queries.
    """

    def __init__(
        self,
        db_path: Path | str = settings.CACHE_DB_PATH,
        similarity_threshold: float = settings.CACHE_SIMILARITY_THRESHOLD,
        embedding_model = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model or get_embedding_model()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    query_embedding BLOB NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, query: str) -> dict | None:
        """Looks up semantically similar cached answers."""
        if not settings.ENABLE_SEMANTIC_CACHE:
            return None

        query_vector = np.array(self.embedding_model.embed_query(query), dtype=np.float32)
        norm_q = np.linalg.norm(query_vector)
        if norm_q == 0:
            return None

        with self._get_connection() as conn:
            cursor = conn.execute("SELECT id, query_text, query_embedding, response_json FROM semantic_cache")
            rows = cursor.fetchall()

        if not rows:
            return None

        best_score = -1.0
        best_payload = None

        for row in rows:
            cached_vector = np.frombuffer(row[2], dtype=np.float32)
            norm_c = np.linalg.norm(cached_vector)
            if norm_c > 0:
                cosine_sim = float(np.dot(query_vector, cached_vector) / (norm_q * norm_c))
                if cosine_sim > best_score:
                    best_score = cosine_sim
                    best_payload = row[3]

        if best_score >= self.similarity_threshold and best_payload:
            cached_data = json.loads(best_payload)
            cached_data["cached"] = True
            cached_data["cache_similarity"] = round(best_score, 4)
            return cached_data

        return None

    def set(self, query: str, response_payload: dict):
        """Saves a query and its synthesized response payload into the cache."""
        if not settings.ENABLE_SEMANTIC_CACHE:
            return

        query_vector = np.array(self.embedding_model.embed_query(query), dtype=np.float32)
        payload_str = json.dumps(response_payload, ensure_ascii=False)

        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO semantic_cache (query_text, query_embedding, response_json, created_at) VALUES (?, ?, ?, ?)",
                (query, query_vector.tobytes(), payload_str, time.time()),
            )
            conn.commit()
