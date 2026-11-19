import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.bm25_service import bm25_service

doc_chunks = [
    "5b8b0197-2a47-46f7-a402-95313fd9bae8_p102_c176",
    "5b8b0197-2a47-46f7-a402-95313fd9bae8_p102_c177",
    "5b8b0197-2a47-46f7-a402-95313fd9bae8_p103_c178",
]

for cid in doc_chunks:
    c = bm25_service.chunk_store.get(cid, {})
    print(f"=== CHUNK: {cid} (p.{c.get('page_number')}) ===")
    print(c.get("chunk_text"))
    print("\n" + "=" * 60 + "\n")

# Check all chunks in Student Handout that mention EJOTY or alphabet positions
print("Searching for 'ejoty' in all chunks:")
found = 0
for cid, c in bm25_service.chunk_store.items():
    text = c.get("chunk_text", "")
    if "ejoty" in text.lower():
        found += 1
        print(f"Found in {cid} (p.{c.get('page_number')}):")
        print(text[:400])
        print("-" * 40)
print(f"Total chunks mentioning ejoty: {found}")
