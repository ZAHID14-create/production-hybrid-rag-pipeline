import sys
import time
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from google import genai
from google.genai import types

api_key = settings.GEMINI_API_KEY.strip()
client = genai.Client(api_key=api_key)

models_to_test = ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.5-flash"]

for model_id in models_to_test:
    print(f"\n--- Testing Model: {model_id} ---", flush=True)
    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model=model_id,
            contents="Say 'Hello World' and nothing else.",
            config=types.GenerateContentConfig(
                max_output_tokens=20,
                temperature=0.0,
            )
        )
        dur = round((time.time() - t0) * 1000, 2)
        txt = resp.text.strip() if resp and resp.text else "(empty)"
        print(f"  SUCCESS in {dur}ms: {txt}", flush=True)
    except Exception as e:
        dur = round((time.time() - t0) * 1000, 2)
        print(f"  FAILED in {dur}ms ({type(e).__name__}): {e}", flush=True)
