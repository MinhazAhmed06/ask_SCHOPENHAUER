from langgraph.graph import StateGraph, END
from .state import RAGState
from .nodes import RAGNodes
from ..cache.semantic_cache import SemanticCache


class RAGWorkflow:
    """Compiled LangGraph Agentic RAG workflow with Self-RAG and CRAG loops."""

    def __init__(self, nodes: RAGNodes | None = None, cache: SemanticCache | None = None):
        self.nodes = nodes or RAGNodes()
        self.cache = cache or SemanticCache()
        self.graph = self._build_graph()

    def _decide_after_route(self, state: RAGState) -> str:
        if state.get("generation"):
            return "end"
        return "retrieve"

    def _decide_after_grading(self, state: RAGState) -> str:
        if state.get("needs_rewrite"):
            return "transform_query"
        return "generate"

    def _build_graph(self):
        builder = StateGraph(RAGState)

        # Add Nodes
        builder.add_node("route_query", self.nodes.route_query)
        builder.add_node("retrieve_and_rerank", self.nodes.retrieve_and_rerank)
        builder.add_node("grade_documents", self.nodes.grade_documents)
        builder.add_node("transform_query", self.nodes.transform_query)
        builder.add_node("generate", self.nodes.generate)
        builder.add_node("verify_groundedness", self.nodes.verify_groundedness)

        # Set Entry Point
        builder.set_entry_point("route_query")

        # Add Edges
        builder.add_conditional_edges(
            "route_query",
            self._decide_after_route,
            {
                "end": END,
                "retrieve": "retrieve_and_rerank",
            },
        )
        builder.add_edge("retrieve_and_rerank", "grade_documents")
        builder.add_conditional_edges(
            "grade_documents",
            self._decide_after_grading,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        builder.add_edge("transform_query", "retrieve_and_rerank")
        builder.add_edge("generate", "verify_groundedness")
        builder.add_edge("verify_groundedness", END)

        return builder.compile()

    def invoke(self, question: str) -> dict:
        # 1. Semantic Cache check
        cached_result = self.cache.get(question)
        if cached_result:
            return cached_result

        # 2. Execute Agentic Graph
        initial_state: RAGState = {
            "question": question,
            "transformed_query": question,
            "documents": [],
            "relevance_scores": [],
            "generation": "",
            "citations": [],
            "retry_count": 0,
            "needs_rewrite": False,
            "is_grounded": True,
            "grounding_explanation": "",
            "cached": False,
            "tokens_used": 0,
        }

        final_state = self.graph.invoke(initial_state)

        result_payload = {
            "question": question,
            "answer": final_state.get("generation", ""),
            "citations": final_state.get("citations", []),
            "is_grounded": final_state.get("is_grounded", True),
            "grounding_explanation": final_state.get("grounding_explanation", ""),
            "tokens_used": final_state.get("tokens_used", 0),
            "retry_count": final_state.get("retry_count", 0),
            "cached": False,
        }

        # 3. Store in Semantic Cache if valid answer generated
        if result_payload["answer"] and result_payload["is_grounded"]:
            self.cache.set(question, result_payload)

        return result_payload
