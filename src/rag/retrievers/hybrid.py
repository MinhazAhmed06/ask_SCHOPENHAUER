from collections import defaultdict
from langchain_core.documents import Document
from ..config import settings
from ..vectorstore.chroma_store import ChromaStore
from ..vectorstore.bm25_store import BM25Store


class HybridRetriever:
    """
    Combines Dense Semantic Search (ChromaDB) and Sparse Lexical Search (BM25)
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_store: ChromaStore | None = None,
        sparse_store: BM25Store | None = None,
        rrf_k: int = settings.RRF_K,
        dense_weight: float = settings.DENSE_WEIGHT,
        sparse_weight: float = settings.SPARSE_WEIGHT,
    ):
        self.dense_store = dense_store or ChromaStore()
        self.sparse_store = sparse_store or BM25Store()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(
        self,
        query: str,
        k: int = settings.RETRIEVAL_TOP_K,
        filter_dict: dict | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Executes dense and sparse searches, then fuses results using RRF.
        Returns top-k documents sorted by combined RRF score.
        """
        dense_results = self.dense_store.similarity_search_with_score(
            query=query,
            k=k * 2,
            filter_dict=filter_dict,
        )
        sparse_results = self.sparse_store.search(query=query, k=k * 2)

        # Document registry by chunk_id or content hash
        doc_map: dict[str, Document] = {}
        rrf_scores: dict[str, float] = defaultdict(float)

        # Score Dense Ranks
        for rank, (doc, sim_score) in enumerate(dense_results, start=1):
            doc_key = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_map[doc_key] = doc
            rrf_scores[doc_key] += self.dense_weight * (1.0 / (self.rrf_k + rank))

        # Score Sparse BM25 Ranks
        for rank, (doc, bm25_score) in enumerate(sparse_results, start=1):
            doc_key = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_map[doc_key] = doc
            rrf_scores[doc_key] += self.sparse_weight * (1.0 / (self.rrf_k + rank))

        # Sort documents by descending RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:k]
        
        max_score = rrf_scores[sorted_keys[0]] if sorted_keys else 1.0
        results: list[tuple[Document, float]] = []
        for key in sorted_keys:
            norm_score = rrf_scores[key] / max_score
            results.append((doc_map[key], float(norm_score)))

        return results
