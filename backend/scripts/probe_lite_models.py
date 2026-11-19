import sys
import time
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from google import genai
from google.genai import types

api_key = settings.GEMINI_API_KEY.strip()
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=10000,
        retry_options=types.HttpRetryOptions(attempts=1),
    )
)

models = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
for m in models:
    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model=m,
            contents="Say 'OK'",
        )
        dur = round((time.time() - t0) * 1000, 2)
        print(f"{m} SUCCESS in {dur}ms: {resp.text.strip()}", flush=True)
    except Exception as e:
        dur = round((time.time() - t0) * 1000, 2)
        code = getattr(e, 'code', None)
        print(f"{m} FAILED in {dur}ms (code={code}): {e}", flush=True)
