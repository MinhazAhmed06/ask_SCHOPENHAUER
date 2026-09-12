from typing import Sequence
import numpy as np
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from ..config import settings


class CrossEncoderReranker:
    """
    Production Cross-Encoder re-ranker.
    
    Evaluates (query, document) pairs through full cross-attention transformer layers,
    achieving significantly higher precision than bi-encoder vector dot products.
    """

    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME, device: str = settings.EMBEDDING_DEVICE):
        self.model_name = model_name
        self.device = device
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        documents: Sequence[Document | tuple[Document, float]],
        top_k: int = settings.RERANK_TOP_K,
    ) -> list[tuple[Document, float]]:
        """
        Scores candidate documents against the exact query and returns the top-k re-ranked list.
        """
        if not documents:
            return []

        doc_list: list[Document] = [
            doc if isinstance(doc, Document) else doc[0] for doc in documents
        ]

        # Prepare (query, doc_text) pairs
        pairs = [[query, doc.page_content] for doc in doc_list]
        raw_scores = self.model.predict(pairs)

        # Sigmoid normalization: 1 / (1 + exp(-score))
        if isinstance(raw_scores, (list, np.ndarray)):
            scores = [1.0 / (1.0 + np.exp(-float(s))) for s in raw_scores]
        else:
            scores = [float(raw_scores)]

        # Pair documents with their re-ranked scores and sort descending
        scored_pairs = list(zip(doc_list, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return scored_pairs[:top_k]
