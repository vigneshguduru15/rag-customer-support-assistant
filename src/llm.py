# src/llm.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"


# ================================
# 1. CORE FUNCTION (keep this)
# ================================
def call_llm(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Groq API Error: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


# ================================
# 2. LLM WRAPPER (NEW)
# ================================
class GroqLLM:
    def predict(self, prompt: str) -> str:
        return call_llm(prompt)


# ================================
# 3. FACTORY FUNCTION (REQUIRED)
# ================================
def get_llm():
    return GroqLLM()