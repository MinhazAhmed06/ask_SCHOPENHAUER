import pickle
import re
from pathlib import Path
from typing import Sequence
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from ..config import settings


class BM25Store:
    """Sparse lexical retriever using BM25 with persistent disk serialization."""

    def __init__(self, index_path: Path | str = settings.BM25_INDEX_PATH):
        self.index_path = Path(index_path)
        self.documents: list[Document] = []
        self.bm25: BM25Okapi | None = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Simple, fast lower-cased alphanumeric tokenizer."""
        return re.findall(r'\w+', text.lower())

    def fit(self, documents: Sequence[Document]):
        """Builds the BM25 index over documents."""
        self.documents = list(documents)
        corpus = [self.tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(corpus)
        self.save()

    def save(self):
        """Saves documents and BM25 structures to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"documents": self.documents, "bm25": self.bm25}, f)

    def load(self) -> bool:
        """Loads index from disk if present."""
        if not self.index_path.exists():
            return False
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.bm25 = data["bm25"]
        return True

    def search(self, query: str, k: int = settings.RETRIEVAL_TOP_K) -> list[tuple[Document, float]]:
        """Returns top-k matching documents with normalized BM25 score."""
        if self.bm25 is None or not self.documents:
            if not self.load():
                return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
        results: list[tuple[Document, float]] = []
        for idx in top_indices:
            if scores[idx] > 0:
                normalized_score = float(scores[idx] / max_score)
                results.append((self.documents[idx], normalized_score))
        return results

    def count(self) -> int:
        if self.bm25 is None:
            self.load()
        return len(self.documents)
