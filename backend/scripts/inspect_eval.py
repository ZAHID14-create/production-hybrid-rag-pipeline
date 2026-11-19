import sys
import json
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.evaluation_service import DATASET_PATH, REPORT_PATH

print("DATASET_PATH exists:", DATASET_PATH.exists(), str(DATASET_PATH))
if DATASET_PATH.exists():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    print("Dataset total questions:", len(d.get("questions", [])))
    if d.get("questions"):
        q0 = d["questions"][0]
        print("Question 0 id:", q0.get("question_id"))
        print("Question 0 relevant chunks:", q0.get("relevant_chunk_ids"))

print("\nREPORT_PATH exists:", REPORT_PATH.exists(), str(REPORT_PATH))
if REPORT_PATH.exists():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        r = json.load(f)
    print("Report total questions evaluated:", r.get("total_questions_evaluated"))
    print("Executive summary key finding:", r.get("executive_summary", {}).get("key_finding"))
    print("\nConfiguration Summaries:")
    for c in r.get("configuration_summaries", []):
        cfg_name = c.get("config_name")
        mrr = c.get("mean_reciprocal_rank")
        ndcg = c.get("ndcg")
        recall = c.get("recall")
        prec = c.get("precision")
        hit = c.get("hit_rate")
        print(f"  [{cfg_name}]")
        print(f"    MRR: {mrr}")
        print(f"    nDCG: {ndcg}")
        print(f"    Recall: {recall}")
        print(f"    Precision: {prec}")
        print(f"    HitRate: {hit}")
