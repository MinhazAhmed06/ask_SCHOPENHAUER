from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from ..config import settings


class EmbeddingFactory:
    """Factory for instantiating embedding models with performance optimizations."""

    _instance: Embeddings | None = None

    @classmethod
    def get_embeddings(cls, model_name: str | None = None, device: str | None = None) -> Embeddings:
        name = model_name or settings.EMBEDDING_MODEL_NAME
        dev = device or settings.EMBEDDING_DEVICE
        
        if cls._instance is None:
            model_kwargs = {"device": dev}
            encode_kwargs = {"normalize_embeddings": True}
            
            cls._instance = HuggingFaceEmbeddings(
                model_name=name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        return cls._instance


def get_embedding_model() -> Embeddings:
    return EmbeddingFactory.get_embeddings()
