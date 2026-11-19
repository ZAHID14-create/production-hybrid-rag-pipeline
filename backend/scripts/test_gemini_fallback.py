"""
Automated Test Suite for Gemini 503 Resilience and Automatic Model Fallback
Tests all required resilience, retry, fallback, status code mapping, and citation integrity invariants.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.models.retrieval import RetrievedChunk
from app.services.generation_service import (
    GenerationService,
    GeminiAuthError,
    GeminiQuotaError,
    GeminiRateLimitError,
    GeminiInvalidRequestError,
    GeminiServiceUnavailableError,
    classify_gemini_error,
)
from google.genai.errors import APIError


def create_sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="vaswani_p1_c1",
            document_id="doc_vaswani",
            document_name="Attention_Is_All_You_Need.pdf",
            page_number=1,
            chunk_index=0,
            chunk_text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality.",
            token_count=45,
            rank=1,
            score=0.88,
            dense_score=0.82,
            dense_rank=1,
            bm25_score=14.5,
            bm25_rank=1,
            rrf_score=0.033,
            reranker_score=0.92,
        ),
        RetrievedChunk(
            chunk_id="spanner_p3_c2",
            document_id="doc_spanner",
            document_name="Spanner_Google_Globally_Distributed_Database.pdf",
            page_number=3,
            chunk_index=2,
            chunk_text="TrueTime API represents time as an interval [earliest, latest] with bounded uncertainty (epsilon). This uncertainty is guaranteed by GPS and atomic clocks.",
            token_count=35,
            rank=2,
            score=0.79,
            dense_score=0.75,
            dense_rank=2,
            bm25_score=11.2,
            bm25_rank=2,
            rrf_score=0.028,
            reranker_score=0.85,
        ),
    ]


def test_1_primary_model_succeeds():
    print("\n--- Test 1: Primary Model Succeeds Normally ---")
    service = GenerationService()
    service.model_name = "gemini-3.8-flash"
    service.fallback_model = "gemini-3.1-flash-lite"

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "The Transformer allows significantly more parallelization [1], while Spanner TrueTime guarantees time intervals [2]."
    mock_client.models.generate_content.return_value = mock_resp

    with patch.object(service, "get_client", return_value=mock_client):
        answer, valid_cits, invalid_cits, sources, gen_time, model_name, has_context, is_fallback, retries, reason = service.generate_answer(
            query="What do Transformer and Spanner achieve?",
            chunks=create_sample_chunks(),
        )

    assert is_fallback is False, f"Expected is_fallback=False, got {is_fallback}"
    assert retries == 0, f"Expected retries=0, got {retries}"
    assert model_name == "gemini-3.8-flash", f"Expected gemini-3.8-flash, got {model_name}"
    assert valid_cits == [1, 2], f"Expected valid citations [1, 2], got {valid_cits}"
    assert len(invalid_cits) == 0, f"Expected 0 invalid citations, got {invalid_cits}"
    assert len(sources) == 2, f"Expected 2 sources, got {len(sources)}"
    assert has_context is True
    print(f"PASS: Primary model succeeded with 0 retries. Model: {model_name}. Valid citations: {valid_cits}")


def test_2_primary_503_retry_succeeds_on_attempt_2():
    print("\n--- Test 2: Primary Model 503 on Attempt 1, Succeeds on Attempt 2 (Backoff & Jitter) ---")
    service = GenerationService()
    service.model_name = "gemini-3.8-flash"
    service.fallback_model = "gemini-3.1-flash-lite"

    # Save original delays to speed up test
    with patch.object(settings, "GEMINI_RETRY_BASE_DELAY", 0.05), patch.object(settings, "GEMINI_RETRY_MAX_DELAY", 0.1):
        mock_client = MagicMock()
        err_503 = Exception("503 This model is currently experiencing high demand")
        success_resp = MagicMock()
        success_resp.text = "The Transformer is a novel architecture [1]."

        # First call fails with 503, second call succeeds
        mock_client.models.generate_content.side_effect = [err_503, success_resp]

        with patch.object(service, "get_client", return_value=mock_client):
            answer, valid_cits, invalid_cits, sources, gen_time, model_name, has_context, is_fallback, retries, reason = service.generate_answer(
                query="What is a Transformer?",
                chunks=create_sample_chunks(),
            )

    assert is_fallback is False, "Expected primary model to succeed on retry without triggering fallback"
    assert retries == 1, f"Expected retries=1, got {retries}"
    assert model_name == "gemini-3.8-flash", f"Expected primary model, got {model_name}"
    assert valid_cits == [1], f"Expected valid citations [1], got {valid_cits}"
    print(f"PASS: Primary model recovered on retry 1 via backoff. Model: {model_name}. Retries: {retries}")


def test_3_primary_repeated_503_fallback_succeeds():
    print("\n--- Test 3: Primary Repeated 503 Overload -> Automatic Fallback Succeeds ---")
    service = GenerationService()
    service.model_name = "gemini-3.8-flash"
    service.fallback_model = "gemini-3.1-flash-lite"

    with patch.object(settings, "GEMINI_RETRY_BASE_DELAY", 0.05), patch.object(settings, "GEMINI_RETRY_MAX_DELAY", 0.1):
        mock_client = MagicMock()

        # Primary model repeatedly fails all attempts with 503/504
        err_503 = Exception("503 The model is currently overloaded. Please try again.")
        err_504 = Exception("504 DEADLINE_EXCEEDED")
        fallback_resp = MagicMock()
        fallback_resp.text = "Spanner TrueTime bounded uncertainty is [earliest, latest] [2]."

        def mock_generate(model, contents, config):
            if model == "gemini-3.8-flash":
                raise err_503
            elif model == "gemini-3.1-flash-lite":
                return fallback_resp
            raise ValueError(f"Unexpected model: {model}")

        mock_client.models.generate_content.side_effect = mock_generate
        mock_client.models.get.return_value = MagicMock(name="models/gemini-3.1-flash-lite")

        with patch.object(service, "get_client", return_value=mock_client):
            answer, valid_cits, invalid_cits, sources, gen_time, model_name, has_context, is_fallback, retries, reason = service.generate_answer(
                query="How does Spanner represent time uncertainty?",
                chunks=create_sample_chunks(),
            )

    assert is_fallback is True, "Expected fallback model to be triggered"
    assert model_name == "gemini-3.1-flash-lite", f"Expected fallback model, got {model_name}"
    assert retries == settings.GEMINI_MAX_RETRIES, f"Expected {settings.GEMINI_MAX_RETRIES} retries on primary before fallback, got {retries}"
    assert valid_cits == [2], f"Expected citation [2], got {valid_cits}"
    assert reason is not None and "gemini-3.8-flash" in reason
    # Ensure source mappings and document names were perfectly preserved
    assert sources[0].document_name == "Attention_Is_All_You_Need.pdf"
    assert sources[1].document_name == "Spanner_Google_Globally_Distributed_Database.pdf"
    print(f"PASS: Seamless fallback to {model_name} after {retries} retries on primary. Reason: {reason}")


def test_4_both_models_fail_service_unavailable():
    print("\n--- Test 4: Both Primary and Fallback Models Fail (HTTP 503) ---")
    service = GenerationService()
    service.model_name = "gemini-3.8-flash"
    service.fallback_model = "gemini-3.1-flash-lite"

    with patch.object(settings, "GEMINI_RETRY_BASE_DELAY", 0.02), patch.object(settings, "GEMINI_RETRY_MAX_DELAY", 0.05):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("503 Service Unavailable: High Demand")
        mock_client.models.get.return_value = MagicMock(name="models/gemini-3.1-flash-lite")

        with patch.object(service, "get_client", return_value=mock_client):
            raised = False
            try:
                service.generate_answer(
                    query="What is TrueTime?",
                    chunks=create_sample_chunks(),
                )
            except GeminiServiceUnavailableError as e:
                raised = True
                assert e.status_code == 503
                assert "Both primary model" in str(e)
                print(f"PASS: Correctly raised GeminiServiceUnavailableError (HTTP {e.status_code}): {e}")

            assert raised, "Expected GeminiServiceUnavailableError when both models fail"


def test_5_invalid_api_key_auth_error():
    print("\n--- Test 5: Invalid API Key / Auth Error (HTTP 401 - Fail Fast) ---")
    service = GenerationService()

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("401 API_KEY_INVALID: API key not valid. Please pass a valid API key.")

    with patch.object(service, "get_client", return_value=mock_client):
        raised = False
        try:
            service.generate_answer(
                query="Test query",
                chunks=create_sample_chunks(),
            )
        except GeminiAuthError as e:
            raised = True
            assert e.status_code == 401
            assert e.is_retryable is False
            # Verify no secret key was exposed in the message
            assert "AIza" not in str(e)
            print(f"PASS: Correctly failed fast on HTTP 401 with GeminiAuthError. Message: {e}")

        assert raised, "Expected GeminiAuthError on 401 error"


def test_6_http_429_quota_exhausted():
    print("\n--- Test 6: HTTP 429 Quota Exhausted (Not Disguised as 503, Fail Fast) ---")
    service = GenerationService()

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric 'Queries' and limit 'Queries per day'")

    with patch.object(service, "get_client", return_value=mock_client):
        raised = False
        try:
            service.generate_answer(
                query="Test query",
                chunks=create_sample_chunks(),
            )
        except GeminiQuotaError as e:
            raised = True
            assert e.status_code == 429
            assert e.is_retryable is False
            print(f"PASS: Correctly distinguished quota limit as GeminiQuotaError (HTTP 429). Message: {e}")

        assert raised, "Expected GeminiQuotaError on quota exhaustion"


def test_7_insufficient_context_handling():
    print("\n--- Test 7: Insufficient Context & Grounding Check ---")
    service = GenerationService()
    service.model_name = "gemini-3.8-flash"

    # Empty chunks
    ans, val_cits, inv_cits, srcs, lat, mod, has_ctx, is_fb, retries, rsn = service.generate_answer(
        query="What is quantum gravity?",
        chunks=[],
    )
    assert has_ctx is False
    assert len(srcs) == 0
    assert "do not contain sufficient information" in ans

    # LLM explicitly states insufficient context
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "The provided documents do not contain sufficient information to answer this question."
    mock_client.models.generate_content.return_value = mock_resp

    with patch.object(service, "get_client", return_value=mock_client):
        ans, val_cits, inv_cits, srcs, lat, mod, has_ctx, is_fb, retries, rsn = service.generate_answer(
            query="What is quantum gravity?",
            chunks=create_sample_chunks(),
        )
    assert has_ctx is False
    print(f"PASS: Insufficient context correctly identified without hallucination.")


if __name__ == "__main__":
    print("======================================================================")
    print("RUNNING AUTOMATED RESILIENCE & MODEL FALLBACK VERIFICATION SUITE")
    print("======================================================================")
    test_1_primary_model_succeeds()
    test_2_primary_503_retry_succeeds_on_attempt_2()
    test_3_primary_repeated_503_fallback_succeeds()
    test_4_both_models_fail_service_unavailable()
    test_5_invalid_api_key_auth_error()
    test_6_http_429_quota_exhausted()
    test_7_insufficient_context_handling()
    print("\n======================================================================")
    print("ALL 7 RESILIENCE & FALLBACK TESTS PASSED PERFECTLY!")
    print("======================================================================")
