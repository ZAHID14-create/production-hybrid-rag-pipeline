"""Reproducible evaluation runner script for the Hybrid RAG pipeline.

Executes multi-configuration evaluation across:
A. BM25 without reranking
B. Dense retrieval without reranking
C. Hybrid RRF without reranking
D. Hybrid RRF with BGE reranking
E. Dense retrieval with BGE reranking
F. BM25 retrieval with BGE reranking

Reports Precision@K, Recall@K, MRR, nDCG@K, HitRate@K, latency distribution,
answer quality, citation correctness, and memory utilization.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.evaluation_service import evaluation_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("evaluation_runner")


def main():
    parser = argparse.ArgumentParser(description="Run Production Hybrid RAG Pipeline Evaluation Benchmark")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 8], help="Cutoff depths for K metrics")
    parser.add_argument("--no-answers", action="store_true", help="Skip answer and citation quality evaluation")
    parser.add_argument("--rerun-live", action="store_true", help="Force live LLM answer generation instead of cached")
    parser.add_argument("--export", type=str, default="", help="Optional custom path to export the JSON report")
    args = parser.parse_args()

    print("=" * 80)
    print("HYBRID RAG PIPELINE: COMPREHENSIVE PHASE 6 EVALUATION BENCHMARK")
    print("=" * 80)
    print(f"K-values evaluated : {args.k_values}")
    print(f"Evaluate Answers   : {not args.no_answers}")
    print(f"Rerun Live LLM     : {args.rerun_live}")
    print("-" * 80)

    report = evaluation_service.run_full_evaluation(
        k_values=args.k_values,
        include_answers=(not args.no_answers),
        use_cached_answers=(not args.rerun_live),
        rerun_live_generation=args.rerun_live,
    )

    print("\n" + "=" * 80)
    print("RETRIEVAL STRATEGY COMPARISON (MACRO-AVERAGED)")
    print("=" * 80)
    header = f"{'Configuration':<28} | {'MRR':<8} | {'nDCG@3':<8} | {'P@3':<8} | {'R@3':<8} | {'Hit@3':<8} | {'Latency (ms)':<12}"
    print(header)
    print("-" * len(header))

    for cfg_key, summ in report.configurations.items():
        p3 = summ.precision_at_k.get(3, 0.0)
        r3 = summ.recall_at_k.get(3, 0.0)
        ndcg3 = summ.ndcg_at_k.get(3, 0.0)
        hit3 = summ.hit_rate_at_k.get(3, 0.0)
        row = f"{summ.display_name:<28} | {summ.mrr:<8.4f} | {ndcg3:<8.4f} | {p3:<8.4f} | {r3:<8.4f} | {hit3:<8.4f} | {summ.mean_latency_ms:<12.2f}"
        print(row)

    print("\n" + "=" * 80)
    print("ANSWER QUALITY & CITATION FAITHFULNESS EVALUATION")
    print("=" * 80)
    for ans in report.answer_evaluations:
        status_flag = "✓ PASS" if (ans.faithfulness_score >= 0.8 and ans.correct_insufficient_context_behavior) else "⚠ WARN"
        print(f"[{status_flag}] ({ans.question_id}) {ans.question[:65]}...")
        print(f"       Category: {ans.category:<22} | Faithfulness: {ans.faithfulness_score:.2f} | Citations: {ans.citations}")
        print(f"       Notes   : {ans.evaluation_notes}")

    print("\n" + "=" * 80)
    print("STAGE-BY-STAGE LATENCY & PERFORMANCE PROFILE")
    print("=" * 80)
    for stage in report.performance.stages:
        print(f" - {stage.stage_name:<38}: Mean={stage.mean_ms:>8.2f} ms | Median={stage.median_ms:>8.2f} ms | P95={stage.p95_ms:>8.2f} ms")

    if report.performance.peak_process_memory_mb:
        print(f"\nPeak Process RSS Memory: {report.performance.peak_process_memory_mb:.2f} MB")

    print("\n" + "=" * 80)
    print(f"BEST PERFORMING RETRIEVAL CONFIGURATION: {report.best_retrieval_configuration}")
    print(f"Rationale: {report.best_configuration_rationale}")
    print("=" * 80)

    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"Exported JSON report to: {export_path.resolve()}")


if __name__ == "__main__":
    main()
