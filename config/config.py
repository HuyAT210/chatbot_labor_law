import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

LLM_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# API Configuration
QWEN_API_KEY = LLM_API_KEY = os.getenv('QWEN_API_KEY', '')
QWEN_API_URL = os.getenv('QWEN_API_URL', 'https://vibe-agent-gateway.eternalai.org/v1')

# Model Configuration
QWEN_MODEL = os.getenv('QWEN_MODEL', 'gpt-4o-mini')

# Use Qwen by default if QWEN_API_KEY is set, otherwise fall back to OpenAI
USE_QWEN = bool(QWEN_API_KEY)

if USE_QWEN and not QWEN_API_KEY:
    raise ValueError("⚠️ Chưa thiết lập QWEN_API_KEY trong .env")
