"""
Script for generating the vector database V_target for the Naive RAG Baseline.

This script processes cleaned Wikipedia articles, applies a naive chunking strategy
(500 characters, no overlap), generates embeddings using the Google Gemini API,
and stores them in a separate ChromaDB collection ('target_base_chunks') for
baseline target-RAG experiments.
"""

import os
import json
import time
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# --- 1. Configuration ---
load_dotenv(override=True)
client = genai.Client()

INPUT_DIR = "../data/wikipedia_articles_cleaned"
DB_PATH = "./chroma_db_wiki"

# --- 2. ChromaDB Initialization ---
print(f"Connecting to existing ChromaDB at {DB_PATH}...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

try:
    chroma_client.delete_collection(name="target_base_chunks")
    print("Old 'target_base_chunks' collection deleted.")
except Exception:
    pass

vector_collection = chroma_client.create_collection(name="target_base_chunks")

# --- 3. Naive Pipeline (500 chars, 0 overlap) ---
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
    chunk_size=500,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=True
)


def get_embeddings_with_retry(texts):
    SUB_BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(texts), SUB_BATCH_SIZE):
        batch = texts[i:i + SUB_BATCH_SIZE]
        while True:
            try:
                response = client.models.embed_content(
                    model='gemini-embedding-001',
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                all_embeddings.extend([emb.values for emb in response.embeddings])
                break
            except Exception as e:
                error_msg = str(e)
                if any(code in error_msg for code in ["429", "503", "500"]):
                    print("API error encountered. Waiting 5 seconds...")
                    time.sleep(5)
                else:
                    raise e
    return all_embeddings


def process_baseline():
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} not found.")
        return

    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"Starting naive baseline ingestion ({len(json_files)} articles)...")

    pbar = tqdm(total=len(json_files), desc="Articles processed")

    for filename in json_files:
        filepath = os.path.join(INPUT_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            article = json.load(f)

        parent_id = str(article.get("oldid", filename))

        full_text = ""
        for section in article.get("sections", []):
            content = section.get("content", "")
            if content:
                full_text += content + "\n\n"

        raw_chunks = text_splitter.split_text(full_text)

        if len(raw_chunks) == 0:
            pbar.update(1)
            continue

        all_chunk_ids = []
        all_chunk_metas = []

        for i, chunk_text in enumerate(raw_chunks):
            all_chunk_ids.append(f"{parent_id}#naive_chunk_{i}")

            all_chunk_metas.append({
                "parent_id": parent_id,
                "chunk_index": i
            })

        try:
            embeddings = get_embeddings_with_retry(raw_chunks)

            vector_collection.add(
                ids=all_chunk_ids,
                embeddings=embeddings,
                documents=raw_chunks,
                metadatas=all_chunk_metas
            )
            time.sleep(0.1)
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

        pbar.update(1)

    pbar.close()
    print("Naive ingestion complete!")


if __name__ == "__main__":
    process_baseline()