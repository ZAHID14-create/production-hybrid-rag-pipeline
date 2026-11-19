import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.models.retrieval import (
    RetrievalMethod,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)
from app.services.bm25_service import bm25_service
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.services.reranker_service import reranker_service

logger = logging.getLogger(__name__)


def compute_rrf_score(ranks: List[int], k: int = 60) -> float:
    """Pure mathematical function computing Reciprocal Rank Fusion (RRF) score.

    Formula:
        RRF(d) = sum(1 / (k + rank_m(d))) for all retrieval systems m in which d appears.

    Args:
        ranks: List of 1-indexed ranks (e.g. [1, 2] if 1st in dense and 2nd in BM25).
        k: Smoothing constant (default 60).

    Returns:
        Combined RRF score float.
    """
    if not ranks:
        return 0.0
    return sum(1.0 / (k + rank) for rank in ranks if rank is not None and rank > 0)


class RetrievalService:
    """Modular retrieval service supporting Dense Vector search, BM25 Lexical search,
    and Hybrid Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        self.default_rrf_k = settings.RRF_K

    def retrieve_dense(
        self,
        query: str,
        top_k: int = 5,
    ) -> Tuple[List[RetrievedChunk], float, int]:
        """Performs semantic dense vector search against ChromaDB using cosine similarity.

        Returns:
            Tuple of (retrieved_chunks, embedding_time_ms, candidate_count)
        """
        if not query.strip():
            return [], 0.0, 0

        # Measure query embedding time separately
        t_emb_0 = time.perf_counter()
        query_vector = embedding_service.embed_query(query)
        embedding_time_ms = round((time.perf_counter() - t_emb_0) * 1000, 2)

        # Retrieve top K from ChromaDB
        raw_results = vector_store.search_by_vector(query_vector, top_k=top_k)

        chunks: List[RetrievedChunk] = []
        for idx, item in enumerate(raw_results, start=1):
            chunks.append(
                RetrievedChunk(
                    rank=idx,
                    chunk_id=item["chunk_id"],
                    document_id=item["document_id"],
                    document_name=item["document_name"],
                    page_number=item["page_number"],
                    chunk_index=item["chunk_index"],
                    chunk_text=item["chunk_text"],
                    token_count=item.get("token_count", 0),
                    score=item["score"],
                    dense_score=item["score"],
                    dense_rank=idx,
                    bm25_score=None,
                    bm25_rank=None,
                    rrf_score=None,
                )
            )

        return chunks, embedding_time_ms, len(raw_results)

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 5,
    ) -> Tuple[List[RetrievedChunk], int]:
        """Performs sparse lexical search using BM25Okapi inverted index.

        Returns:
            Tuple of (retrieved_chunks, candidate_count)
        """
        if not query.strip():
            return [], 0

        raw_results = bm25_service.search(query, top_k=top_k)

        chunks: List[RetrievedChunk] = []
        for idx, item in enumerate(raw_results, start=1):
            chunks.append(
                RetrievedChunk(
                    rank=idx,
                    chunk_id=item["chunk_id"],
                    document_id=item["document_id"],
                    document_name=item["document_name"],
                    page_number=item["page_number"],
                    chunk_index=item["chunk_index"],
                    chunk_text=item["chunk_text"],
                    token_count=item.get("token_count", 0),
                    score=round(item["score"], 4),
                    dense_score=None,
                    dense_rank=None,
                    bm25_score=round(item["score"], 4),
                    bm25_rank=idx,
                    rrf_score=None,
                )
            )

        return chunks, len(raw_results)

    def retrieve_hybrid_rrf(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: Optional[int] = None,
        candidate_depth: Optional[int] = None,
    ) -> Tuple[List[RetrievedChunk], float, int, int]:
        """Performs Hybrid Retrieval combining Dense Vector Search and BM25 Lexical Search
        using Reciprocal Rank Fusion (RRF).

        Formula:
            RRF(d) = sum(1 / (k + rank_m(d))) for m in {dense, bm25}

        Returns:
            Tuple of (top_k_chunks, embedding_time_ms, dense_candidates_count, bm25_candidates_count)
        """
        if not query.strip():
            return [], 0.0, 0, 0

        k = rrf_k if rrf_k is not None and rrf_k > 0 else self.default_rrf_k
        # Candidate pool depth: retrieve at least 3x top_k or 20 candidates per channel
        fetch_depth = candidate_depth or max(top_k * 3, 20)

        # 1. Dense retrieval candidates
        t_emb_0 = time.perf_counter()
        query_vector = embedding_service.embed_query(query)
        embedding_time_ms = round((time.perf_counter() - t_emb_0) * 1000, 2)
        dense_candidates = vector_store.search_by_vector(query_vector, top_k=fetch_depth)

        # 2. BM25 lexical retrieval candidates
        bm25_candidates = bm25_service.search(query, top_k=fetch_depth)

        # 3. Fuse candidates using chunk_id
        # Map: chunk_id -> dict of accumulated metrics
        fused_pool: Dict[str, Dict[str, Any]] = {}

        # Process Dense candidates
        for rank, item in enumerate(dense_candidates, start=1):
            cid = item["chunk_id"]
            if cid not in fused_pool:
                fused_pool[cid] = {
                    "chunk_id": cid,
                    "document_id": item["document_id"],
                    "document_name": item["document_name"],
                    "page_number": item["page_number"],
                    "chunk_index": item["chunk_index"],
                    "chunk_text": item["chunk_text"],
                    "token_count": item.get("token_count", 0),
                    "dense_rank": rank,
                    "dense_score": item["score"],
                    "bm25_rank": None,
                    "bm25_score": None,
                    "ranks": [rank],
                }
            else:
                fused_pool[cid]["dense_rank"] = rank
                fused_pool[cid]["dense_score"] = item["score"]
                fused_pool[cid]["ranks"].append(rank)

        # Process BM25 candidates
        for rank, item in enumerate(bm25_candidates, start=1):
            cid = item["chunk_id"]
            if cid not in fused_pool:
                fused_pool[cid] = {
                    "chunk_id": cid,
                    "document_id": item["document_id"],
                    "document_name": item["document_name"],
                    "page_number": item["page_number"],
                    "chunk_index": item["chunk_index"],
                    "chunk_text": item["chunk_text"],
                    "token_count": item.get("token_count", 0),
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": rank,
                    "bm25_score": round(item["score"], 4),
                    "ranks": [rank],
                }
            else:
                fused_pool[cid]["bm25_rank"] = rank
                fused_pool[cid]["bm25_score"] = round(item["score"], 4)
                fused_pool[cid]["ranks"].append(rank)

        # 4. Calculate RRF scores
        for cid, entry in fused_pool.items():
            entry["rrf_score"] = compute_rrf_score(entry["ranks"], k=k)

        # 5. Sort candidates descending by RRF score, with stable tie-breaking
        ranked_candidates = list(fused_pool.values())
        ranked_candidates.sort(
            key=lambda c: (
                round(c["rrf_score"], 8),
                c["dense_score"] or 0.0,
                c["bm25_score"] or 0.0,
                c["chunk_id"],
            ),
            reverse=True,
        )

        # 6. Take top K
        top_candidates = ranked_candidates[:top_k]

        chunks: List[RetrievedChunk] = []
        for final_rank, entry in enumerate(top_candidates, start=1):
            chunks.append(
                RetrievedChunk(
                    rank=final_rank,
                    chunk_id=entry["chunk_id"],
                    document_id=entry["document_id"],
                    document_name=entry["document_name"],
                    page_number=entry["page_number"],
                    chunk_index=entry["chunk_index"],
                    chunk_text=entry["chunk_text"],
                    token_count=entry["token_count"],
                    score=round(entry["rrf_score"], 6),
                    dense_score=entry["dense_score"],
                    dense_rank=entry["dense_rank"],
                    bm25_score=entry["bm25_score"],
                    bm25_rank=entry["bm25_rank"],
                    rrf_score=round(entry["rrf_score"], 6),
                )
            )

        return chunks, embedding_time_ms, len(dense_candidates), len(bm25_candidates)

    def search(self, request: RetrievalRequest) -> RetrievalResponse:
        """Unified retrieval entrypoint supporting BM25, Dense, Hybrid RRF, and BGE Reranking."""
        t_start = time.perf_counter()
        method = request.retrieval_method.value if hasattr(request.retrieval_method, "value") else str(request.retrieval_method)
        query = request.query.strip()
        rerank_enabled = bool(request.rerank)

        # When reranking is requested, pull a tuned candidate pool to re-order (default 10)
        if rerank_enabled:
            fetch_k = max(request.top_k, (request.reranker_top_k or request.top_k), settings.RERANKER_CANDIDATE_POOL)
        else:
            fetch_k = request.top_k

        embedding_time_ms: Optional[float] = None
        dense_count: Optional[int] = None
        bm25_count: Optional[int] = None
        rrf_k_used: Optional[int] = None
        reranking_time_ms: Optional[float] = None
        reranker_model_used: Optional[str] = None

        if not query:
            return RetrievalResponse(
                query=query,
                retrieval_method=method,
                total_results=0,
                results=[],
                execution_time_ms=0.0,
                rerank_enabled=rerank_enabled,
            )

        if method == RetrievalMethod.DENSE.value:
            results, embedding_time_ms, dense_count = self.retrieve_dense(query, top_k=fetch_k)
        elif method == RetrievalMethod.BM25.value:
            results, bm25_count = self.retrieve_bm25(query, top_k=fetch_k)
        elif method == RetrievalMethod.HYBRID.value:
            rrf_k_used = request.rrf_k if request.rrf_k is not None else self.default_rrf_k
            results, embedding_time_ms, dense_count, bm25_count = self.retrieve_hybrid_rrf(
                query, top_k=fetch_k, rrf_k=rrf_k_used
            )
        else:
            raise ValueError(f"Unsupported retrieval method: '{method}'")

        # Mark original rank before any optional reranking
        for chunk in results:
            chunk.original_rank = chunk.rank

        # Apply BGE Cross-Encoder Reranking if enabled
        if rerank_enabled:
            target_final_k = request.reranker_top_k or request.top_k
            if results:
                results, reranking_time_ms = reranker_service.rerank(
                    query=query,
                    candidates=results,
                    top_k=target_final_k,
                    batch_size=request.reranker_batch_size,
                )
            else:
                reranking_time_ms = 0.0
            reranker_model_used = reranker_service.model_name
        else:
            # If reranking is disabled, truncate to requested top_k
            results = results[:request.top_k]

        total_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return RetrievalResponse(
            query=query,
            retrieval_method=method,
            total_results=len(results),
            results=results,
            execution_time_ms=total_elapsed_ms,
            embedding_time_ms=embedding_time_ms,
            dense_candidates_count=dense_count,
            bm25_candidates_count=bm25_count,
            rrf_k=rrf_k_used,
            rerank_enabled=rerank_enabled,
            reranking_time_ms=reranking_time_ms,
            reranker_model=reranker_model_used,
        )


retrieval_service = RetrievalService()
