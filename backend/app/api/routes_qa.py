import time
from fastapi import APIRouter, HTTPException, status

from app.models.qa import QARequest, QAResponse
from app.models.retrieval import RetrievalRequest
from app.services.retrieval_service import retrieval_service
from app.services.generation_service import (
    generation_service,
    GeminiAuthError,
    GeminiQuotaError,
    GeminiRateLimitError,
    GeminiInvalidRequestError,
    GeminiServiceUnavailableError,
)

router = APIRouter(prefix="/qa", tags=["Question Answering"])


@router.post(
    "/ask",
    response_model=QAResponse,
    status_code=status.HTTP_200_OK,
    summary="End-to-End Grounded Q&A with Source Citations",
    description="Retrieves multi-channel candidates (BM25, Dense, Hybrid RRF), optionally reranks with BGE, and synthesizes a grounded answer with exact page citations using Google Gemini with automatic model fallback."
)
def ask_question(request: QARequest) -> QAResponse:
    t_start = time.perf_counter()
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question query cannot be empty or whitespace.",
        )

    # 1. Execute Retrieval & Optional Reranking
    retrieval_req = RetrievalRequest(
        query=clean_query,
        retrieval_method=request.retrieval_method,
        top_k=request.top_k,
        rrf_k=request.rrf_k,
        rerank=request.rerank,
        reranker_top_k=request.reranker_top_k,
    )

    try:
        retrieval_resp = retrieval_service.search(retrieval_req)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval pipeline failed: {str(e)}",
        )

    # 2. Select Candidate Chunks for Context
    context_chunks = retrieval_resp.results

    # 3. Call Gemini Generation Service with Citations & Fallback
    try:
        (
            answer,
            valid_citations,
            invalid_citations,
            sources,
            generation_time_ms,
            model_name,
            has_sufficient_context,
            is_fallback_used,
            retry_count,
            fallback_reason,
        ) = generation_service.generate_answer(
            query=clean_query,
            chunks=context_chunks,
            temperature=request.temperature,
        )
    except GeminiAuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )
    except (GeminiQuotaError, GeminiRateLimitError) as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )
    except GeminiInvalidRequestError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except GeminiServiceUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error occurred during grounded answer generation: {str(e)}",
        )

    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return QAResponse(
        query=clean_query,
        answer=answer,
        retrieval_method=retrieval_resp.retrieval_method,
        rerank_enabled=retrieval_resp.rerank_enabled,
        citations=valid_citations,
        invalid_citations=invalid_citations,
        sources=sources,
        has_sufficient_context=has_sufficient_context,
        total_execution_time_ms=total_time_ms,
        retrieval_time_ms=retrieval_resp.execution_time_ms,
        embedding_time_ms=retrieval_resp.embedding_time_ms,
        reranking_time_ms=retrieval_resp.reranking_time_ms,
        generation_time_ms=generation_time_ms,
        model_name=model_name,
        is_fallback_used=is_fallback_used,
        retry_count=retry_count,
        fallback_reason=fallback_reason,
    )
