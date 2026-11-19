import sys
import io
import re
import time
from pathlib import Path

# Alias re to regex for transformers
sys.modules['regex'] = re

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi import UploadFile
from app.api.routes_documents import upload_document
from app.services.index_manager import index_manager

# First clean up any temporary test chunks if present
index_manager.delete_document_indices("test_doc_attn")

docs_to_upload = [
    ("documents/attention_is_all_you_need.pdf", "attention_is_all_you_need.pdf"),
    ("documents/spanner_distributed_database.pdf", "spanner_distributed_database.pdf"),
]

for file_path_str, filename in docs_to_upload:
    file_path = Path(file_path_str)
    if not file_path.exists():
        print(f"[ERROR] {file_path} not found!", flush=True)
        continue

    print(f"\n=======================================================", flush=True)
    print(f"[INFO] Ingesting '{filename}' ({file_path.stat().st_size:,} bytes) via upload_document route handler...", flush=True)
    t0 = time.time()
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    upload_file = UploadFile(
        file=io.BytesIO(file_bytes),
        filename=filename,
        headers={"content-type": "application/pdf"}
    )

    resp = upload_document(file=upload_file)
    elapsed = time.time() - t0
    doc = resp.document
    print(f"[SUCCESS] Ingested in {elapsed:.2f}s", flush=True)
    print(f"  Message        : {resp.message}", flush=True)
    print(f"  Document ID    : {doc.document_id}", flush=True)
    print(f"  Document Name  : {doc.document_name}", flush=True)
    print(f"  Total Pages    : {doc.total_pages}", flush=True)
    print(f"  Pages w/ Text  : {doc.pages_with_text}", flush=True)
    print(f"  Chunks Created : {doc.total_chunks}", flush=True)
    print(f"  Tokens Extracted: {doc.total_tokens:,}", flush=True)

print("\n" + "=" * 60, flush=True)
print("[INFO] Synchronized Index Statistics:", flush=True)
stats = index_manager.get_stats()
print(f"  ChromaDB Chunks: {stats.chroma_chunk_count}")
print(f"  BM25 Chunks    : {stats.bm25_chunk_count}")
print(f"  Synchronized   : {stats.is_synchronized}")
print(f"  Total Chunks   : {stats.total_indexed_chunks}")

status = index_manager.get_status()
print(f"  Total Documents: {status.total_documents}")
print(f"  Last Action    : {status.last_action}")
