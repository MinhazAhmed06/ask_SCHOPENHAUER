from typing import Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter

from .state import RAGState
from ..config import settings
from ..retrievers.hybrid import HybridRetriever
from ..retrievers.reranker import CrossEncoderReranker
from ..retrievers.query_transform import QueryTransformer
from ..guardrails.token_budget import TokenBudgetManager
from ..guardrails.hallucination import GroundingVerifier


class RAGNodes:
    """Nodes containing pure execution logic for the LangGraph RAG state machine."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        transformer: QueryTransformer | None = None,
        budgeter: TokenBudgetManager | None = None,
        verifier: GroundingVerifier | None = None,
        llm: ChatOpenRouter | None = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.transformer = transformer or QueryTransformer()
        self.budgeter = budgeter or TokenBudgetManager()
        self.verifier = verifier or GroundingVerifier()
        self.llm = llm or ChatOpenRouter(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENROUTER_API_KEY,
        )

    def route_query(self, state: RAGState) -> dict[str, Any]:
        """Classify if the input is a simple greeting/meta question or requires corpus retrieval."""
        question = state["question"].strip().lower()
        greetings = {"hello", "hi", "hey", "who are you", "what can you do", "help"}
        
        if question in greetings or len(question) < 5:
            return {
                "generation": "Greetings! I am the Arthur Schopenhauer Philosophical RAG assistant. Ask me anything about Schopenhauer's essays, 'The Wisdom of Life', his metaphysics, ethics, suicide, personality, or doctrine of the Will.",
                "documents": [],
                "citations": [],
                "is_grounded": True,
                "needs_rewrite": False,
            }
        return {"transformed_query": state["question"], "retry_count": state.get("retry_count", 0)}

    def retrieve_and_rerank(self, state: RAGState) -> dict[str, Any]:
        """Runs Hybrid (Dense + BM25) search and Cross-Encoder re-ranking."""
        query = state.get("transformed_query") or state["question"]
        
        # 1. Hybrid candidates
        candidates = self.retriever.retrieve(query=query, k=settings.RETRIEVAL_TOP_K)
        
        # 2. Cross-Encoder re-ranking
        reranked = self.reranker.rerank(query=query, documents=candidates, top_k=settings.RERANK_TOP_K)
        
        # 3. Fit to prompt token budget
        budgeted_docs = self.budgeter.fit_documents_to_budget(reranked)
        
        scores = [score for _, score in reranked[: len(budgeted_docs)]]
        
        return {
            "documents": budgeted_docs,
            "relevance_scores": scores,
        }

    def grade_documents(self, state: RAGState) -> dict[str, Any]:
        """Evaluates document relevance signal; flags for query rewrite if signal is weak."""
        scores = state.get("relevance_scores", [])
        retry_count = state.get("retry_count", 0)
        
        if (not scores or max(scores) < settings.MIN_DOC_RELEVANCE_SCORE) and retry_count < settings.MAX_RETRY_LOOPS:
            return {"needs_rewrite": True}
        return {"needs_rewrite": False}

    def transform_query(self, state: RAGState) -> dict[str, Any]:
        """Corrective RAG (CRAG) node: Rewrites the question for improved vector recall."""
        current_query = state.get("transformed_query") or state["question"]
        queries = self.transformer.generate_multi_queries(current_query, num_queries=2)
        new_query = queries[1] if len(queries) > 1 else current_query
        return {
            "transformed_query": new_query,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def format_context_and_citations(self, documents: list[Document]) -> tuple[str, list[dict]]:
        context_parts = []
        citations = []
        for i, doc in enumerate(documents, start=1):
            meta = doc.metadata
            cite_id = f"[{i}]"
            section = meta.get("section", "General")
            chapter = meta.get("chapter", "Unknown")
            book_page = meta.get("book_pages", meta.get("book_page", "N/A"))
            pdf_page = meta.get("pdf_pages", meta.get("pdf_page", "N/A"))
            
            context_parts.append(
                f"--- SOURCE {cite_id} (Section: {section}, Chapter: {chapter}, Book Page: {book_page}) ---\n{doc.page_content}"
            )
            citations.append({
                "source_id": cite_id,
                "section": section,
                "chapter": chapter,
                "book_page": book_page,
                "pdf_page": pdf_page,
                "snippet": doc.page_content[:200] + "...",
            })
        return "\n\n".join(context_parts), citations

    def generate(self, state: RAGState) -> dict[str, Any]:
        """Synthesizes the answer strictly grounded in retrieved documents with in-text citations."""
        documents = state.get("documents", [])
        question = state["question"]
        
        if not documents:
            return {
                "generation": "I could not find relevant passages in Arthur Schopenhauer's essays to answer your question accurately.",
                "citations": [],
                "is_grounded": True,
                "grounding_explanation": "No documents retrieved.",
            }

        formatted_context, citations = self.format_context_and_citations(documents)

        prompt = ChatPromptTemplate.from_template(
            """You are an authoritative scholar and expert on Arthur Schopenhauer's philosophical works.
Answer the user's question based STRICTLY and ONLY on the provided context passages and their header metadata.

Guidelines:
1. Provide a direct, thorough, and eloquent explanation reflecting Schopenhauer's exact arguments.
2. Embed citation markers like [1], [2] at the end of sentences referring to specific sources.
3. If the context does not contain enough information to answer, state honestly: "The provided Schopenhauer writings do not contain information to answer this question."
4. Do NOT speculate or invent philosophical doctrines outside the context.

Context Passages:
{context}

Question:
{question}

Scholarly Answer:"""
        )

        chain = prompt | self.llm | StrOutputParser()
        generation = chain.invoke({"context": formatted_context, "question": question}).strip()

        total_tokens = self.budgeter.count_tokens(formatted_context) + self.budgeter.count_tokens(generation)

        return {
            "generation": generation,
            "citations": citations,
            "tokens_used": total_tokens,
        }

    def verify_groundedness(self, state: RAGState) -> dict[str, Any]:
        """Self-RAG Verifier: Validates faithfulness of output against context including metadata headers."""
        documents = state.get("documents", [])
        generation = state.get("generation", "")
        question = state.get("question", "")

        if not documents or not generation:
            return {"is_grounded": True, "grounding_explanation": "Skipped verification (no docs/generation)."}

        formatted_context, _ = self.format_context_and_citations(documents)
        grounding_result = self.verifier.verify(question=question, response=generation, context=formatted_context)

        return {
            "is_grounded": grounding_result.is_grounded,
            "grounding_explanation": grounding_result.explanation,
        }
