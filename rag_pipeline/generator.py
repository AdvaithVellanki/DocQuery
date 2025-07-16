# rag_pipeline/generator.py

import requests


def query_llm(model: str, prompt: str) -> str:
    """Send prompt to local Ollama instance and return the generated response."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    return response.json().get("response", "").strip()
