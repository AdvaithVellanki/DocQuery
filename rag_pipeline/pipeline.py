# rag_pipeline/pipeline.py

from rag_pipeline.retriever import retrieve, load_chunks
from rag_pipeline.generator import query_llm
from rag_pipeline.prompts import build_prompt


def answer_query(query: str, model="mistral", top_k=3) -> str:
    chunks = load_chunks()
    results = retrieve(query, top_k=top_k)
    selected_chunks = [chunks[r["chunk"]] for r in results]
    prompt = build_prompt(selected_chunks, query)
    response = query_llm(model, prompt)
    return response


if __name__ == "__main__":
    q = "How do I use await with asyncio in Python?"
    answer = answer_query(q, model="mistral")
    print(f"🔍 Question: {q}\n\n🧠 Answer:\n{answer}")
