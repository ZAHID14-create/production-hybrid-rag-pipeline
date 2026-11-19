"""
Live 5-Question Evaluation Script for Production Hybrid RAG Pipeline with Resilience Fallback.
Executes 5 live queries against the indexed corpus and measures:
- Actual Model Used
- Fallback Triggered (True/False)
- Retry Count
- Response Latencies (Embedding, Retrieval, Rerank, Gemini Generation, Total)
- Valid Citations & Source Grounding
"""

import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.models.qa import QARequest
from app.services.retrieval_service import retrieval_service
from app.services.generation_service import generation_service
from app.models.retrieval import RetrievalRequest, RetrievalMethod

QUESTIONS = [
    "How does the TrueTime API represent time uncertainty in Spanner?",
    "What are the key advantages of Transformers over recurrent neural networks?",
    "How does Reciprocal Rank Fusion combine rankings from multiple retrieval channels?",
    "What are the components and equations of the Multi-Head Attention mechanism?",
    "What role does Paxos play in Spanner's consensus and replication architecture?",
]

def run_live_qa_suite():
    print("=" * 80)
    print("EXECUTING LIVE 5-QUESTION PIPELINE WITH AUTOMATIC GEMINI FALLBACK")
    print("=" * 80)

    results = []

    for idx, query in enumerate(QUESTIONS, start=1):
        print(f"\n[{idx}/5] Query: {query}")
        t_start = time.perf_counter()

        # Step 1: Execute Hybrid Retrieval + BGE Rerank
        retrieval_req = RetrievalRequest(
            query=query,
            retrieval_method=RetrievalMethod.HYBRID,
            top_k=5,
            rrf_k=60,
            rerank=True,
            reranker_top_k=3,
        )

        retrieval_resp = retrieval_service.search(retrieval_req)
        context_chunks = retrieval_resp.results

        # Step 2: Grounded Generation with Gemini Resilience & Fallback
        (
            answer,
            valid_citations,
            invalid_citations,
            sources,
            gen_time_ms,
            model_name,
            has_context,
            is_fallback,
            retries,
            reason,
        ) = generation_service.generate_answer(
            query=query,
            chunks=context_chunks,
            temperature=0.2,
        )

        total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

        res_summary = {
            "query": query,
            "answer_preview": answer[:220] + "..." if len(answer) > 220 else answer,
            "model_used": model_name,
            "is_fallback_used": is_fallback,
            "retries": retries,
            "fallback_reason": reason,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "sources_count": len(sources),
            "sources": [f"{s.document_name} (p.{s.page_number})" for s in sources],
            "has_context": has_context,
            "retrieval_time_ms": retrieval_resp.execution_time_ms,
            "embedding_time_ms": retrieval_resp.embedding_time_ms,
            "reranking_time_ms": retrieval_resp.reranking_time_ms,
            "generation_time_ms": gen_time_ms,
            "total_time_ms": total_time_ms,
        }
        results.append(res_summary)

        print(f"  -> Model: {model_name} (Fallback: {is_fallback}, Retries: {retries})")
        print(f"  -> Latency: Gen={gen_time_ms}ms | Total={total_time_ms}ms | Retrieval={retrieval_resp.execution_time_ms}ms")
        print(f"  -> Citations: {valid_citations} (Sources: {len(sources)})")
        print(f"  -> Sample Answer: {res_summary['answer_preview']}")

    print("\n" + "=" * 80)
    print("LIVE 5-QUESTION SUMMARY REPORT")
    print("=" * 80)
    for idx, r in enumerate(results, start=1):
        print(f"\nQuestion {idx}: {r['query']}")
        print(f"  Model: {r['model_used']} | Fallback: {r['is_fallback_used']} | Retries: {r['retries']}")
        print(f"  Latencies: Gen: {r['generation_time_ms']} ms | Total: {r['total_time_ms']} ms")
        print(f"  Citations: {r['valid_citations']} (from {r['sources']})")
        print(f"  Answer: {r['answer_preview']}")

    return results

if __name__ == "__main__":
    run_live_qa_suite()
