import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.bm25_service import bm25_service
from app.services.retrieval_service import retrieval_service
from app.models.retrieval import RetrievalRequest, RetrievalMethod

print("Total chunks in BM25:", len(bm25_service.chunk_ids))
ejoty_chunks = [cid for cid, c in bm25_service.chunk_store.items() if "ejoty" in c.get("text", "").lower()]
print("Chunks containing EJOTY:", len(ejoty_chunks))
for cid in ejoty_chunks:
    c = bm25_service.chunk_store[cid]
    print(f"Chunk: {cid} | Doc: {c.get('document_name')} | Page: {c.get('page_number')}")
    print(f"Text snippet:\n{c.get('text', '')[:300]}\n")

query = "Explain the concept of EJOTY"
print(f"--- EXECUTING RETRIEVAL FOR: '{query}' ---")
req = RetrievalRequest(
    query=query,
    retrieval_method=RetrievalMethod.HYBRID,
    top_k=10,
    rrf_k=60,
    rerank=True,
    reranker_top_k=5,
)
resp = retrieval_service.search(req)
print(f"Total results: {len(resp.results)}")
for r in resp.results:
    print(f"Rank {r.rank} | Chunk: {r.chunk_id} | Doc: {r.document_name} (p.{r.page_number})")
    print(f"  Dense score: {r.dense_score} (dense_rank: {r.dense_rank})")
    print(f"  BM25 score: {r.bm25_score} (bm25_rank: {r.bm25_rank})")
    print(f"  RRF score: {r.rrf_score} (orig_rank: {r.original_rank})")
    print(f"  BGE Reranker score: {r.reranker_score}")
    print(f"  Snippet: {r.chunk_text[:180]}...\n")
