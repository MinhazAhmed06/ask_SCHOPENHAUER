import time
from typing import Sequence
from .dataset import EvalSample, BENCHMARK_DATASET
from ..graph.workflow import RAGWorkflow


class RAGEvaluator:
    """Evaluates the RAG system performance across Precision, Recall, Faithfulness, and Latency."""

    def __init__(self, workflow: RAGWorkflow | None = None):
        self.workflow = workflow or RAGWorkflow()

    def evaluate_sample(self, sample: EvalSample) -> dict:
        start_time = time.perf_counter()
        result = self.workflow.invoke(sample.question)
        latency_ms = (time.perf_counter() - start_time) * 1000

        citations = result.get("citations", [])
        answer = result.get("answer", "")
        is_grounded = result.get("is_grounded", True)

        # 1. Context Recall (did we retrieve the target chapter?)
        retrieved_chapters = [c.get("chapter", "").lower() for c in citations]
        target_chap_lower = sample.expected_chapter.lower()
        context_recall = 1.0 if any(target_chap_lower in chap or chap in target_chap_lower for chap in retrieved_chapters) else 0.0

        # 2. Context Precision (fraction of expected keywords found in retrieved snippets)
        all_snippets = " ".join(c.get("snippet", "").lower() for c in citations)
        matched_keywords = [kw for kw in sample.expected_keywords if kw.lower() in all_snippets]
        context_precision = len(matched_keywords) / len(sample.expected_keywords) if sample.expected_keywords else 1.0

        # 3. Answer Keyword Coverage
        answer_lower = answer.lower()
        answer_kw_matches = [kw for kw in sample.expected_keywords if kw.lower() in answer_lower]
        answer_relevancy = len(answer_kw_matches) / len(sample.expected_keywords) if sample.expected_keywords else 1.0

        # 4. Faithfulness score
        faithfulness = 1.0 if is_grounded else 0.5

        return {
            "id": sample.id,
            "question": sample.question,
            "expected_chapter": sample.expected_chapter,
            "context_recall": context_recall,
            "context_precision": round(context_precision, 2),
            "faithfulness": faithfulness,
            "answer_relevancy": round(answer_relevancy, 2),
            "tokens_used": result.get("tokens_used", 0),
            "latency_ms": round(latency_ms, 2),
            "cached": result.get("cached", False),
        }

    def run_benchmark(self, dataset: Sequence[EvalSample] = BENCHMARK_DATASET) -> dict:
        results = []
        for sample in dataset:
            res = self.evaluate_sample(sample)
            results.append(res)

        avg_recall = sum(r["context_recall"] for r in results) / len(results) if results else 0.0
        avg_precision = sum(r["context_precision"] for r in results) / len(results) if results else 0.0
        avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results) if results else 0.0
        avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results) if results else 0.0
        avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0.0

        summary = {
            "total_samples": len(results),
            "average_context_recall": round(avg_recall, 3),
            "average_context_precision": round(avg_precision, 3),
            "average_faithfulness": round(avg_faithfulness, 3),
            "average_answer_relevancy": round(avg_relevancy, 3),
            "average_latency_ms": round(avg_latency, 2),
            "sample_results": results,
        }
        return summary
