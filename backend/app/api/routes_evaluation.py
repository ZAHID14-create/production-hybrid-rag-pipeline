import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status

from app.models.evaluation import (
    EvaluationDataset,
    EvaluationQuestion,
    EvaluationReportResponse,
    EvaluationRunRequest,
)
from app.services.evaluation_service import evaluation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["RAG Evaluation & Benchmarking"])


@router.post(
    "/run",
    response_model=EvaluationReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute comprehensive RAG evaluation and benchmarking",
    description="Runs reproducible evaluation across 6 retrieval configurations, evaluates answer faithfulness and citation accuracy, and measures stage latencies.",
)
def run_evaluation(request: Optional[EvaluationRunRequest] = None) -> EvaluationReportResponse:
    """Triggers the full multi-configuration evaluation and returns a complete report."""
    req = request or EvaluationRunRequest()
    try:
        report = evaluation_service.run_full_evaluation(
            k_values=req.k_values,
            include_answers=req.include_answer_evaluation,
            use_cached_answers=req.use_cached_answers,
            rerun_live_generation=req.rerun_live_generation,
        )
        return report
    except Exception as e:
        logger.error("Evaluation run failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.get(
    "/results",
    response_model=EvaluationReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve evaluation report",
    description="Returns the persisted evaluation benchmark report without re-running models. Supports mode='benchmark' (fixed baseline) or mode='live' (live repository).",
)
def get_latest_results(mode: str = "benchmark") -> EvaluationReportResponse:
    """Fetches the benchmark report or live evaluation report."""
    if mode == "benchmark":
        return evaluation_service.get_benchmark_baseline_report()

    report = evaluation_service.get_latest_report()
    if report is None:
        logger.info("No prior live evaluation report found. Generating an initial live benchmark report...")
        report = evaluation_service.run_full_evaluation()
    return report


@router.get(
    "/dataset",
    response_model=EvaluationDataset,
    status_code=status.HTTP_200_OK,
    summary="Get evaluation benchmark dataset",
    description="Returns all questions in the evaluation dataset along with verified status and relevance judgments.",
)
def get_evaluation_dataset() -> EvaluationDataset:
    """Returns the ground-truth evaluation dataset."""
    return evaluation_service.get_dataset()


@router.post(
    "/dataset/add",
    response_model=EvaluationDataset,
    status_code=status.HTTP_201_CREATED,
    summary="Add a question to the evaluation dataset",
    description="Extends the benchmark dataset with a new ground-truth or candidate question.",
)
def add_evaluation_question(question: EvaluationQuestion) -> EvaluationDataset:
    """Appends an evaluation question to the dataset."""
    try:
        updated_dataset = evaluation_service.add_question(question)
        return updated_dataset
    except Exception as e:
        logger.error("Failed to add question to evaluation dataset: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add question: {str(e)}",
        )
