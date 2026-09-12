from typing import Sequence
import uuid
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..config import settings


class SmartChunker:
    """
    Production chunking engine supporting recursive splitting and parent-document linkage.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        separators: list[str] = settings.SEPARATORS,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def estimate_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split()) * 4 // 3

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        """
        Splits page or chapter documents into chunks, enriching each chunk with:
        - unique chunk_id
        - chunk_index
        - token_count
        - parent_doc_id
        - preserved source & chapter metadata
        """
        chunks: list[Document] = []
        
        for doc_idx, doc in enumerate(documents):
            parent_id = doc.metadata.get("doc_id", f"doc_{doc_idx}_{uuid.uuid4().hex[:8]}")
            sub_chunks = self.splitter.split_text(doc.page_content)
            
            for chunk_idx, chunk_text in enumerate(sub_chunks):
                if not chunk_text.strip():
                    continue
                    
                chunk_metadata = dict(doc.metadata)
                chunk_id = f"{parent_id}_c{chunk_idx}"
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "total_chunks_in_parent": len(sub_chunks),
                    "parent_id": parent_id,
                    "token_count": self.estimate_tokens(chunk_text),
                    "char_count": len(chunk_text),
                })
                
                chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
                
        return chunks
