import sys
import re
import json
from pathlib import Path

# Provide standard re module alias for transformers
sys.modules['regex'] = re

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.models.retrieval import RetrievalMethod, RetrievalRequest
from app.services.retrieval_service import retrieval_service
from app.services.generation_service import generation_service

queries = [
    {
        "id": "Q1_CROSS_DOC",
        "title": "Cross-Document Architectural Synthesis (Transformer d_model vs. MiniLM)",
        "query": "How does the input dimensionality of the Transformer model in Vaswani et al. compare with the vector dimension produced by the all-MiniLM-L6-v2 embedding model in our RAG specification?",
        "expected_domain": "Cross-Document (Vaswani et al. + RAG Tech Spec)",
    },
    {
        "id": "Q2_MULTI_PASSAGE_TF",
        "title": "Deep Learning Multi-Passage Synthesis (Transformers)",
        "query": "Why does the Transformer architecture use sinusoidal positional encodings, and what mathematical function prevents the dot-product attention logits from growing excessively large in high dimensions?",
        "expected_domain": "Vaswani et al. (Multi-Passage)",
    },
    {
        "id": "Q3_MULTI_PASSAGE_SPANNER",
        "title": "Distributed Systems Multi-Passage Synthesis (Cloud Infrastructure)",
        "query": "How does Spanner's TrueTime API represent time uncertainty, and how do Paxos consensus groups use this to achieve externally consistent read-only transactions without locking?",
        "expected_domain": "Spanner OSDI 2012 (Multi-Passage)",
    },
    {
        "id": "Q4_RETRIEVAL_DISCRIMINATION",
        "title": "Lexical vs. Semantic Retrieval Discrimination (2PC over Paxos)",
        "query": "What mechanism coordinates two-phase commit over replicated state machines in distributed transaction execution?",
        "expected_domain": "Spanner OSDI 2012",
    },
    {
        "id": "Q5_NEGATIVE_CONTROL",
        "title": "Out-of-Domain Strict Negative Control (CRISPR-Cas9)",
        "query": "What is the catalytic mechanism of the CRISPR-Cas9 enzyme in targeted genome editing?",
        "expected_domain": "Out-of-Domain (Zero Ground Truth)",
    },
]

methods = [
    ("BM25", RetrievalMethod.BM25, False),
    ("Dense", RetrievalMethod.DENSE, False),
    ("Hybrid RRF", RetrievalMethod.HYBRID, False),
    ("Hybrid + BGE Rerank", RetrievalMethod.HYBRID, True),
]

print("=" * 80)
print("EXPANDED CORPUS RETRIEVAL & QUESTION ANSWERING VERIFICATION")
print(f"Total Indexed Documents: 4 | Total Chunks: 79")
print("=" * 80)

results_summary = []

for item in queries:
    qid = item["id"]
    title = item["title"]
    q = item["query"]
    print(f"\n--------------------------------------------------------------------------------")
    print(f"[{qid}] {title}")
    print(f"Query: \"{q}\"")
    print("-" * 80)

    # 1. Compare retrieval methods
    retrieval_comparisons = {}
    for name, method, rerank in methods:
        req = RetrievalRequest(
            query=q,
            retrieval_method=method,
            top_k=3,
            rerank=rerank,
            reranker_top_k=3,
        )
        resp = retrieval_service.search(req)
        retrieval_comparisons[name] = [
            {
                "rank": r.rank,
                "chunk_id": r.chunk_id,
                "doc_name": r.document_name,
                "page": r.page_number,
                "score": round(r.score, 4),
                "reranker_score": round(r.reranker_score, 4) if r.reranker_score is not None else None,
                "preview": r.chunk_text[:90].replace("\n", " "),
            }
            for r in resp.results[:3]
        ]
        top1 = resp.results[0] if resp.results else None
        top1_info = f"Rank 1: [{top1.document_name} P.{top1.page_number}] score={top1.score:.4f}" if top1 else "No results"
        if rerank and top1 and top1.reranker_score is not None:
            top1_info += f" (reranker_raw={top1.reranker_score:.4f})"
        print(f"  {name:<20}: {top1_info} | Time: {resp.execution_time_ms:.2f}ms")

    # 2. Run through Gemini Generation with Hybrid + BGE Context
    hybrid_bge_req = RetrievalRequest(
        query=q,
        retrieval_method=RetrievalMethod.HYBRID,
        top_k=5,
        rerank=True,
        reranker_top_k=3,
    )
    search_resp = retrieval_service.search(hybrid_bge_req)
    
    # Generate answer with retry on transient 503
    max_retries = 4
    ans_text = ""
    citations = []
    invalid_cits = []
    sources = []
    gen_ms = 0.0
    model_name = "gemini"
    has_ctx = False

    for attempt in range(max_retries):
        try:
            ans_text, citations, invalid_cits, sources, gen_ms, model_name, has_ctx = generation_service.generate_answer(
                query=q,
                chunks=search_resp.results,
            )
            break
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                wait_sec = 4 * (attempt + 1)
                print(f"  [WARN] Gemini 503 temporary demand spike. Retrying in {wait_sec}s (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(wait_sec)
            else:
                print(f"  [ERROR] Gemini call failed: {e}", flush=True)
                ans_text = f"Error: {e}"
                break

    print(f"\n  [GEMINI GENERATION (Model: {model_name}, Time: {gen_ms:.2f}ms)]")
    print(f"  Has Context : {has_ctx}")
    print(f"  Citations   : {citations} (Invalid: {invalid_cits})")
    print(f"  Answer Text :\n  {ans_text.strip()}\n")
    print(f"  Cited Sources:")
    for s in sources:
        print(f"    Source [{s.citation_id}]: {s.document_name} (Page {s.page_number}) - ID: {s.chunk_id}")

    results_summary.append({
        "question_id": qid,
        "title": title,
        "query": q,
        "retrieval_comparisons": retrieval_comparisons,
        "answer_text": ans_text,
        "citations": citations,
        "sources": [s.model_dump() for s in sources],
        "has_context": has_ctx,
        "generation_time_ms": gen_ms,
    })

# Save results to a json file
out_path = Path("data/evaluation/expanded_corpus_test_results.json")
out_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
print("\n" + "=" * 80)
print(f"Full results persisted to {out_path.resolve()}")
print("=" * 80)
