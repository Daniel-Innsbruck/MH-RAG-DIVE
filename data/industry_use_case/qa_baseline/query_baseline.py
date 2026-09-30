import sys
import os
from pathlib import Path
import csv
import json
import time
from dotenv import load_dotenv
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from connectors.k5_client import K5RAGClient
from common.config import GenConfig

load_dotenv()

config = GenConfig()

# ==========================================
# Config
# ==========================================
INPUT_CSV = "single_hop_qa_set.csv"
OUTPUT_JSONL = "baseline_rag_results.jsonl"

# --- RAG API Config ---
RAG_BASE_URL = config.rag_base_url
INTERNAL_TOKEN = config.rag_api_key
SERVICE_NAME = "aio"
TENANT_NAME = "evalrag"

MAX_RETRIES = 3


def process_baseline():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: File '{INPUT_CSV}' not found in current working directory.")
        return

    client = K5RAGClient(
        base_url=RAG_BASE_URL,
        internal_token=INTERNAL_TOKEN,
        service_name=SERVICE_NAME,
        tenant_name=TENANT_NAME,
        timeout=60
    )

    open(OUTPUT_JSONL, 'w').close()

    print(f"Reading questions from '{INPUT_CSV}' and sending them to K5-RAG-System...")

    with open(INPUT_CSV, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        for index, row in enumerate(reader):
            question = row.get("question", "")
            reference_answer = row.get("answer", "")

            if not question:
                continue

            print(f"[{index + 1}] Question: {question[:60]}...")

            for attempt in range(MAX_RETRIES):
                session_id = None
                try:
                    session_id = client.get_thread_id()

                    rag_result = client.ask_rag(question, session_id)

                    eval_record = {
                        "conversation": [
                            {
                                "question": question,
                                "answer": reference_answer,
                                "rag_answer": rag_result.get("answer", ""),
                                "context": rag_result.get("context", []),
                                "multi_hop_flag": 0
                            }
                        ]
                    }

                    with open(OUTPUT_JSONL, mode="a", encoding="utf-8") as outfile:
                        outfile.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

                    break

                except Exception as e:
                    print(f"Error for question {index + 1} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2)
                    else:
                        print(f"Failed to process question {index + 1} after {MAX_RETRIES} attempts.")

                finally:
                    if session_id:
                        client.delete_thread(session_id)

            time.sleep(10)

    print(f"\nDone! Results are in '{OUTPUT_JSONL}'.")


if __name__ == "__main__":
    process_baseline()