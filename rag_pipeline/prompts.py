# rag_pipeline/retriever.py


def build_prompt(context_chunks, user_query):
    """Build a prompt with retrieved context and the user's question."""
    context = "\n\n".join(context_chunks)
    return f"""
You are a helpful assistant trained on Python's official documentation.

Use the following context to answer the user's question.

Context:
{context}

Question: {user_query}
Answer:
""".strip()
