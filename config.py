# config.py - loads everything from .env so other modules
# don't have to deal with env vars directly

import os
from dotenv import load_dotenv

load_dotenv()

# -- LLM backend: "openrouter" | "groq" | "ollama" | "gemini"
LLM_BACKEND = os.getenv("LLM_BACKEND", "openrouter")

# openrouter (what we ended up using - fast, cheap, parallel-friendly)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

# groq - tried this first but hit the 500k/day token cap
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ollama local fallback - too slow on my 4gb vram card tbh
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# gemini - rate limits were brutal, kept as backup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# -- paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUBSET_PATH = os.path.join(DATA_DIR, "enron_subset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
EXTRACTIONS_DIR = os.path.join(OUTPUT_DIR, "extractions")
GRAPH_DB_PATH = os.path.join(OUTPUT_DIR, "memory_graph.db")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXTRACTIONS_DIR, exist_ok=True)

# -- extraction settings
EXTRACTION_VERSION = os.getenv("EXTRACTION_VERSION", "v1.0")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
BATCH_SIZE = 5  # emails per batch
# tuned these delays per backend to avoid getting rate-limited
_RATE_DELAYS = {"openrouter": 0.3, "groq": 2.0, "ollama": 0.5, "gemini": 4.5}
RATE_LIMIT_DELAY = _RATE_DELAYS.get(LLM_BACKEND, 1.0)

# -- neo4j (running in docker: neo4j:5-community)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "layer10memory")
