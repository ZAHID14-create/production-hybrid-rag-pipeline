import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.models.retrieval import RetrievalRequest, RetrievalMethod
from app.services.retrieval_service import retrieval_service
from app.services.generation_service import generation_service

query = "Explain the concept of EJOTY"
req = RetrievalRequest(
    query=query,
    retrieval_method=RetrievalMethod.HYBRID,
    top_k=5,
    rrf_k=60,
    rerank=True,
    reranker_top_k=3,
)
retrieval_resp = retrieval_service.search(req)

print("Retrieved Chunks passed to Gemini:")
for idx, r in enumerate(retrieval_resp.results, 1):
    print(f"\n--- Source [{idx}] (Chunk {r.chunk_id}, Page {r.page_number}) ---")
    print(f"Rerank Score: {r.reranker_score}")
    print(r.chunk_text)

(
    answer,
    valid_citations,
    invalid_citations,
    sources,
    gen_time,
    model_name,
    has_context,
    is_fallback,
    retries,
    reason,
) = generation_service.generate_answer(
    query=query,
    chunks=retrieval_resp.results,
    temperature=0.2,
)

print("\n" + "=" * 80)
print("GENERATED ANSWER:")
print("=" * 80)
print(answer)
print(f"\nModel: {model_name} | Fallback: {is_fallback} | Retries: {retries}")
print(f"Valid Citations: {valid_citations}")
