# rag_pipeline/retriever.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Chunking and saving documents
# This script processes raw text documents, splits them into smaller chunks,
RAW_DIR = "data/docs_raw"
CHUNKED_DIR = "data/docs_cleaned"
os.makedirs(CHUNKED_DIR, exist_ok=True)


def load_raw_docs():
    docs = []
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(RAW_DIR, filename), "r", encoding="utf-8") as f:
                text = f.read()
                docs.append((filename, text))
    return docs


def chunk_and_save():
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = load_raw_docs()

    for name, text in docs:
        chunks = splitter.split_text(text)
        output_path = os.path.join(CHUNKED_DIR, name)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"--- chunk {i} ---\n{chunk}\n\n")
        print(f"[✔] Chunked {name} into {len(chunks)} pieces.")


# Embeddings + FAISS index creation
EMBEDDING_DIR = "data/embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)


def load_chunks():
    """Load all chunks from cleaned docs."""
    chunks = []
    sources = []
    for filename in os.listdir(CHUNKED_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(CHUNKED_DIR, filename), "r", encoding="utf-8") as f:
                current_chunk = []
                for line in f:
                    if line.startswith("--- chunk"):
                        if current_chunk:
                            chunks.append("".join(current_chunk))
                            sources.append(filename)
                            current_chunk = []
                    else:
                        current_chunk.append(line)
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    sources.append(filename)
    return chunks, sources


def embed_and_save():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks, sources = load_chunks()
    embeddings = model.encode(chunks, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, os.path.join(EMBEDDING_DIR, "faiss.index"))

    with open(os.path.join(EMBEDDING_DIR, "sources.pkl"), "wb") as f:
        pickle.dump(sources, f)

    print(f"[✔] Saved FAISS index with {len(chunks)} chunks.")


# Retriever logic to return top-k relevant chunks
def retrieve(query, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(os.path.join(EMBEDDING_DIR, "faiss.index"))
    with open(os.path.join(EMBEDDING_DIR, "sources.pkl"), "rb") as f:
        sources = pickle.load(f)

    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), top_k)

    results = []
    for idx in I[0]:
        results.append({"chunk": idx, "source": sources[idx]})
    return results


if __name__ == "__main__":
    chunk_and_save()
    embed_and_save()
    q = "How do I create a coroutine in asyncio?"
    retrieved = retrieve(q)
    print(f"Top results for '{q}':\n", retrieved)
