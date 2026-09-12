import tiktoken
from langchain_core.documents import Document
from ..config import settings


class TokenBudgetManager:
    """
    Manages prompt token economics, context window budgeting,
    and prevents LLM prompt overflow.
    """

    def __init__(self, max_tokens: int = settings.MAX_TOKENS_PER_REQUEST):
        self.max_tokens = max_tokens
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text.split()) * 4 // 3

    def fit_documents_to_budget(
        self,
        documents: list[Document | tuple[Document, float]],
        prompt_overhead_tokens: int = 400,
        max_context_tokens: int | None = None,
    ) -> list[Document]:
        """
        Dynamically trims retrieved documents to ensure total prompt tokens
        stay safely within the allocated budget.
        """
        if max_context_tokens is not None:
            budget = max_context_tokens
        else:
            budget = max(0, self.max_tokens - prompt_overhead_tokens)

        allocated_docs: list[Document] = []
        current_tokens = 0

        for item in documents:
            doc = item[0] if isinstance(item, tuple) else item
            doc_tokens = self.count_tokens(doc.page_content)
            
            if current_tokens + doc_tokens <= budget:
                allocated_docs.append(doc)
                current_tokens += doc_tokens
            else:
                remaining_tokens = budget - current_tokens
                if remaining_tokens > 20:
                    approx_chars = remaining_tokens * 4
                    truncated_content = doc.page_content[:approx_chars] + "..."
                    truncated_meta = dict(doc.metadata)
                    truncated_meta["truncated"] = True
                    allocated_docs.append(Document(page_content=truncated_content, metadata=truncated_meta))
                break

        return allocated_docs
