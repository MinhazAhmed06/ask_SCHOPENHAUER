from pathlib import Path
from typing import Sequence
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from ..config import settings
from ..embeddings.factory import get_embedding_model


class ChromaStore:
    """Production persistent ChromaDB wrapper for dense semantic search."""

    def __init__(
        self,
        persist_directory: Path | str = settings.CHROMA_PERSIST_DIR,
        collection_name: str = settings.CHROMA_COLLECTION_NAME,
        embedding_model = None,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model or get_embedding_model()
        
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
        )

    def add_documents(self, documents: Sequence[Document], batch_size: int = 250) -> int:
        """Indexes documents into ChromaDB in robust batches with chunk IDs."""
        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            ids = [doc.metadata.get("chunk_id") for doc in batch]
            if any(id_ is None for id_ in ids):
                ids = None
            self.vector_store.add_documents(documents=batch, ids=ids)
        return total

    def similarity_search_with_score(
        self,
        query: str,
        k: int = settings.RETRIEVAL_TOP_K,
        filter_dict: dict | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Returns list of (Document, cosine_similarity_score).
        Langchain Chroma returns distance (lower is closer); we convert to normalized similarity score.
        """
        results_with_distance = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict,
        )
        scored_results: list[tuple[Document, float]] = []
        for doc, distance in results_with_distance:
            # Cosine distance: similarity = 1.0 - (distance / 2.0) or 1.0 / (1.0 + distance)
            similarity = 1.0 / (1.0 + max(0.0, float(distance)))
            scored_results.append((doc, similarity))
        return scored_results

    def count(self) -> int:
        return self.vector_store._collection.count()
