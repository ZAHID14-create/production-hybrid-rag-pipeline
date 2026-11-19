import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from google import genai

api_key = settings.GEMINI_API_KEY.strip()
if not api_key:
    print("API Key is missing or empty.")
    sys.exit(1)

client = genai.Client(api_key=api_key)
print("Listing available models from official google-genai client...")
try:
    models = list(client.models.list())
    supported = []
    for m in models:
        # m.name is usually "models/gemini-..."
        name = getattr(m, 'name', '') or str(m)
        supported_actions = getattr(m, 'supported_actions', []) or getattr(m, 'supported_generation_methods', [])
        supported.append((name, supported_actions))
    
    print(f"Total models retrieved: {len(supported)}")
    flash_models = [name for name, actions in supported if 'flash' in name.lower()]
    print("Flash models available:")
    for f in flash_models:
        print("  -", f)
except Exception as e:
    print("Error listing models:", type(e), e)
