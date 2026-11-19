import sys
import time
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from google import genai
from google.genai import types

api_key = settings.GEMINI_API_KEY.strip()
# Disable SDK-level retries so our custom retry & fallback logic has full control!
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=10000,  # 10,000 ms = 10s
        retry_options=types.HttpRetryOptions(attempts=1), # 1 attempt = no retry
    )
)

models = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash"]
for m in models:
    print(f"Testing {m} ...", flush=True)
    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model=m,
            contents="Say 'OK'",
        )
        dur = round((time.time() - t0) * 1000, 2)
        print(f"  {m} SUCCESS in {dur}ms: {resp.text.strip()}", flush=True)
    except Exception as e:
        dur = round((time.time() - t0) * 1000, 2)
        code = getattr(e, 'code', None)
        status = getattr(e, 'status', None)
        print(f"  {m} FAILED in {dur}ms ({type(e).__name__}, code={code}, status={status}): {e}", flush=True)
