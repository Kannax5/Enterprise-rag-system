# app/generation/llm_client.py
import requests

LLM_ENDPOINT = "http://localhost:11434/api/generate"

def generate(prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
    response = requests.post(LLM_ENDPOINT, json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }, timeout=180)
    response.raise_for_status()
    return response.json().get("response", "").strip()