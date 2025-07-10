# rag_pipeline/retriever.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

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


if __name__ == "__main__":
    chunk_and_save()
