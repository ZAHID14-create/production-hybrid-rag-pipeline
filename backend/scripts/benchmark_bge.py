"""
BGE Performance Optimization Benchmark Script.
Measures baseline vs. optimizations independently:
1. Baseline: FP32, default threads, max_length=512, candidate pool=10
2. Thread tuning: torch.set_num_threads(2, 4, default)
3. Max length: 384 vs 512
4. INT8 dynamic quantization: torch.quantization.quantize_dynamic vs FP32
5. Candidate pool depth: 8 vs 10 vs 15

Evaluates ranking preservation and latency across multiple queries.
"""

import sys
import time
import copy
from pathlib import Path
from typing import List, Dict, Any, Tuple

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.services.retrieval_service import retrieval_service
from app.models.retrieval import RetrievalRequest, RetrievalMethod, RetrievedChunk

BENCHMARK_QUERIES = [
    "Explain the concept of EJOTY",
    "What are the properties of twin alternate series?",
    "How does Multi-Head Attention work?",
    "What is the TrueTime API bounded uncertainty?",
]

def load_raw_model():
    model_name = "BAAI/bge-reranker-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, local_files_only=True, use_safetensors=True
    )
    model.eval()
    return tokenizer, model

def score_pairs_custom(
    model,
    tokenizer,
    query: str,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 16,
) -> Tuple[List[float], float]:
    t0 = time.perf_counter()
    pairs = [[query, t] for t in texts]
    all_scores = []
    
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            outputs = model(**encoded, return_dict=True)
            logits = outputs.logits
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            elif logits.dim() > 1 and logits.size(-1) > 1:
                logits = logits[:, -1]
            logits = logits.view(-1).float()
            probs = torch.sigmoid(logits).cpu().tolist()
            if isinstance(probs, (float, int)):
                probs = [float(probs)]
            all_scores.extend(probs)
            
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return all_scores, latency_ms

def run_benchmarks():
    print("=" * 80)
    print("STARTING BGE RERANKER OPTIMIZATION BENCHMARK")
    print("=" * 80)
    
    tokenizer, base_model = load_raw_model()
    
    # 1. Fetch real retrieved candidates for each test query from the live index
    query_candidates: Dict[str, List[RetrievedChunk]] = {}
    for q in BENCHMARK_QUERIES:
        req = RetrievalRequest(
            query=q,
            retrieval_method=RetrievalMethod.HYBRID,
            top_k=15,
            rerank=False,
        )
        res = retrieval_service.search(req)
        query_candidates[q] = res.results
        print(f"Query: '{q}' -> Retrieved {len(res.results)} candidates from index.")

    # EXPERIMENT 1: Thread tuning on FP32 baseline
    print("\n--- EXPERIMENT 1: CPU Thread Tuning (FP32, max_length=512, pool=10) ---")
    thread_configs = [None, 2, 4, 6]
    for tc in thread_configs:
        if tc is not None:
            torch.set_num_threads(tc)
        latencies = []
        for q in BENCHMARK_QUERIES:
            texts = [c.chunk_text for c in query_candidates[q][:10]]
            _, lat = score_pairs_custom(base_model, tokenizer, q, texts, max_length=512)
            latencies.append(lat)
        avg_lat = sum(latencies) / len(latencies)
        print(f"Threads: {tc if tc else 'Default (' + str(torch.get_num_threads()) + ')'} -> Avg Latency: {avg_lat:.2f} ms")

    # EXPERIMENT 2: Sequence Length (384 vs 512, with 4 threads)
    print("\n--- EXPERIMENT 2: Sequence Length (FP32, threads=4, pool=10) ---")
    torch.set_num_threads(4)
    for ml in [512, 384, 256]:
        latencies = []
        top1_matches = 0
        for q in BENCHMARK_QUERIES:
            texts = [c.chunk_text for c in query_candidates[q][:10]]
            scores, lat = score_pairs_custom(base_model, tokenizer, q, texts, max_length=ml)
            latencies.append(lat)
        avg_lat = sum(latencies) / len(latencies)
        print(f"Max Length: {ml} -> Avg Latency: {avg_lat:.2f} ms")

    # EXPERIMENT 3: INT8 Dynamic Quantization
    print("\n--- EXPERIMENT 3: INT8 Dynamic Quantization vs FP32 Baseline (threads=4, max_length=384, pool=10) ---")
    quantized_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(base_model),
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    
    # Compare FP32 vs INT8 latencies and ranking fidelity
    fp32_lats, int8_lats = [], []
    ranking_agreements = []
    
    for q in BENCHMARK_QUERIES:
        texts = [c.chunk_text for c in query_candidates[q][:10]]
        
        scores_fp32, lat_fp32 = score_pairs_custom(base_model, tokenizer, q, texts, max_length=384)
        scores_int8, lat_int8 = score_pairs_custom(quantized_model, tokenizer, q, texts, max_length=384)
        
        fp32_lats.append(lat_fp32)
        int8_lats.append(lat_int8)
        
        # Check ranking order
        rank_fp32 = sorted(range(len(scores_fp32)), key=lambda i: scores_fp32[i], reverse=True)
        rank_int8 = sorted(range(len(scores_int8)), key=lambda i: scores_int8[i], reverse=True)
        
        top1_match = rank_fp32[0] == rank_int8[0]
        top3_overlap = len(set(rank_fp32[:3]).intersection(set(rank_int8[:3])))
        ranking_agreements.append((top1_match, top3_overlap))
        
        print(f"Query: '{q[:35]}...'")
        print(f"  FP32: {lat_fp32:.1f} ms | INT8: {lat_int8:.1f} ms (Speedup: {lat_fp32/max(lat_int8, 0.1):.2f}x)")
        print(f"  Top 1 Match: {top1_match} | Top 3 Overlap: {top3_overlap}/3")
        
    avg_fp32 = sum(fp32_lats) / len(fp32_lats)
    avg_int8 = sum(int8_lats) / len(int8_lats)
    print(f"\nOverall FP32 Avg: {avg_fp32:.2f} ms | INT8 Avg: {avg_int8:.2f} ms (Speedup: {avg_fp32/avg_int8:.2f}x)")

    # EXPERIMENT 4: Candidate Pool Depth (INT8, threads=4, max_length=384)
    print("\n--- EXPERIMENT 4: Candidate Pool Depth (INT8, threads=4, max_length=384) ---")
    for pool_size in [8, 10, 15]:
        lats = []
        for q in BENCHMARK_QUERIES:
            texts = [c.chunk_text for c in query_candidates[q][:pool_size]]
            _, lat = score_pairs_custom(quantized_model, tokenizer, q, texts, max_length=384)
            lats.append(lat)
        print(f"Pool size {pool_size} candidates -> Avg Latency: {sum(lats)/len(lats):.2f} ms")

if __name__ == "__main__":
    run_benchmarks()
