import time
import psutil
import requests


def query_ollama(model: str, prompt: str):
    start_time = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    elapsed = time.time() - start_time
    output = response.json().get("response", "")
    token_count = len(output.split())  # crude estimate
    return elapsed, token_count, output[:100] + "..."


def log_memory_usage():
    process = psutil.Process()
    return round(process.memory_info().rss / (1024 * 1024), 2)


if __name__ == "__main__":
    models = ["mistral", "llama3"]
    prompt = "Explain how Python's asyncio works with an example."

    for model in models:
        print(f"\nBenchmarking model: {model}")
        mem_before = log_memory_usage()
        elapsed, tokens, preview = query_ollama(model, prompt)
        mem_after = log_memory_usage()
        print(f"Latency: {elapsed:.2f} sec")
        print(f"Tokens generated: {tokens}")
        print(f"Speed: {tokens / elapsed:.2f} tok/sec")
        print(f"Memory used: {mem_after - mem_before:.2f} MB")
        print(f"Output preview: {preview}")
