from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter
from ..config import settings


class QueryTransformer:
    """
    Transforms and expands user queries into optimized search representations:
    1. Multi-Query Expansion: Generates diverse keyword & conceptual perspectives.
    2. HyDE (Hypothetical Document Embeddings): Synthesizes a hypothetical answer to align embeddings.
    """

    def __init__(self, llm: ChatOpenRouter | None = None):
        self.llm = llm or ChatOpenRouter(
            model=settings.LLM_MODEL,
            temperature=0.2,
            api_key=settings.OPENROUTER_API_KEY,
        )

    def generate_multi_queries(self, question: str, num_queries: int = 3) -> list[str]:
        """Generates multiple search queries from different angles."""
        prompt = ChatPromptTemplate.from_template(
            """You are an AI language model assistant for a search retrieval system.
Your task is to generate {num_queries} different versions of the given user query to retrieve relevant documents from a philosophical vector database.
By generating multiple perspectives on the user question, your goal is to help overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Do not number them or add any other text.

Original query: {question}"""
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            raw_output = chain.invoke({"question": question, "num_queries": num_queries})
            queries = [q.strip() for q in raw_output.strip().split("\n") if q.strip()]
            if question not in queries:
                queries.insert(0, question)
            return queries[: num_queries + 1]
        except Exception:
            return [question]

    def generate_hyde_document(self, question: str) -> str:
        """Generates a hypothetical philosophical answer passage to use for embedding lookup."""
        prompt = ChatPromptTemplate.from_template(
            """Write a concise, hypothetical excerpt from Arthur Schopenhauer's essays that would directly answer the following question.
Do not preface the excerpt; write purely in Schopenhauer's philosophical style and voice.

Question: {question}
Excerpt:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({"question": question}).strip()
        except Exception:
            return question
