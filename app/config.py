"""Configuration and environment settings for the Aster & Row agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge-base"
DATA_DIR = BASE_DIR / "data"
ORDERS_FILE = DATA_DIR / "orders.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
