import sys
import os
import glob
from pathlib import Path
import json
import time
import requests
import logging
import smtplib
import traceback
from email.mime.text import MIMEText
from datetime import datetime
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from connectors.rag_client import RAGClient
from common.config import GenConfig

load_dotenv()
config = GenConfig()

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_JSONL = "clapnq_dev_answerable_orig.jsonl"
OUTPUT_DIR = "baseline_results"

RAG_BASE_URL = config.rag_base_url
INTERNAL_TOKEN = config.rag_api_key
CONFIG_API_URL = f"{RAG_BASE_URL.rstrip('/')}/config"

HEADERS = {
    "X-API-Key": os.getenv("API_KEY", "DEIN_PASSWORT_HIER"),
    "Content-Type": "application/json"
}

# Email settings from .env
SMTP_SERVER = os.getenv("GEN_SMTP_SERVER", "smtp.uibk.ac.at")
SMTP_PORT = int(os.getenv("GEN_SMTP_PORT", 587))
SMTP_USER = os.getenv("GEN_SMTP_USER")
SMTP_PASS = os.getenv("GEN_SMTP_PASS")
SMTP_FROM = os.getenv("GEN_SMTP_FROM")
ALERT_EMAIL = os.getenv("GEN_ALERT_EMAIL")

MAX_RETRIES = 3

EXPERIMENTS = {
    "rag_setting_standard": {
        "k_chunks": 5,
        "chroma_collection_name": "target_base_chunks_gemini",
        "llm_model": "gemma-4-31b-it",
        "embedding_model": "models/gemini-embedding-001",
        "system_prompt": "You are a helpful assistant. Answer the question based solely on the provided context and the previous chat history. If the context does not contain the answer, state that you cannot find the answer. Never answer questions with your internal knowledge.",
        "history_mode": "all",
        "history_n": 3,
        "strategy": "baseline"
    },
    "rag_setting_chunksize_8": {
        "k_chunks": 8,
        "chroma_collection_name": "target_base_chunks_gemini",
        "llm_model": "gemma-4-31b-it",
        "embedding_model": "models/gemini-embedding-001",
        "system_prompt": "You are a helpful assistant. Answer the question based solely on the provided context and the previous chat history.",
        "history_mode": "all",
        "history_n": 3,
        "strategy": "baseline"
    },
    "rag_setting_chunksize_1": {
        "k_chunks": 1,
        "chroma_collection_name": "target_base_chunks_gemini",
        "llm_model": "gemma-4-31b-it",
        "embedding_model": "models/gemini-embedding-001",
        "system_prompt": "You are a helpful assistant. Answer the question based solely on the provided context and the previous chat history.",
        "history_mode": "all",
        "history_n": 3,
        "strategy": "baseline"
    },
    "rag_setting_embedding_model_bge_small": {
        "k_chunks": 5,
        "chroma_collection_name": "target_base_chunks_bge",
        "llm_model": "gemma-4-31b-it",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "system_prompt": "You are a helpful assistant. Answer the question based solely on the provided context and the previous chat history.",
        "history_mode": "all",
        "history_n": 3,
        "strategy": "baseline"
    },
    "rag_setting_embedding_model_all-MiniLM": {
        "k_chunks": 5,
        "chroma_collection_name": "target_base_chunks_minilm",
        "llm_model": "gemma-4-31b-it",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "system_prompt": "You are a helpful assistant. Answer the question based solely on the provided context and the previous chat history.",
        "history_mode": "all",
        "history_n": 3,
        "strategy": "baseline"
    },
    "rag_setting_standard_with_agent": {
        "k_chunks": 5,
        "chroma_collection_name": "target_base_chunks_gemini",
        "llm_model": "gemma-4-31b-it",
        "embedding_model": "models/gemini-embedding-001",
        "system_prompt": "You are a helpful assistant. Answer the question based solely on the provided context and the previous chat history.",
        "history_mode": "all",
        "history_n": 3,
        "strategy": "agentic"
    }
}

# ==========================================
# SETUP LOGGING
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
master_log_file = os.path.join(OUTPUT_DIR, f"run_{run_timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(master_log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def send_email_notification(subject, message):
    """Sends an email notification if credentials are provided in .env."""
    if not all([SMTP_USER, SMTP_PASS, SMTP_FROM, ALERT_EMAIL]):
        logger.warning("Email credentials not fully set in .env. Skipping email notification.")
        return

    try:
        msg = MIMEText(message)
        msg["Subject"] = f"[RAG Pipeline] {subject}"
        msg["From"] = SMTP_FROM
        msg["To"] = ALERT_EMAIL

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        logger.info(f"Email notification sent: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")


def extract_reference_answer(item):
    """Extracts the reference answer from ClapNQ based on byte offsets."""
    try:
        annotations = item.get("annotations", [])
        if annotations and "passage_answer" in annotations[0]:
            ans_idx = annotations[0]["passage_answer"].get("candidate_index", -1)
            if ans_idx != -1:
                candidates = item.get("passage_answer_candidates", [])
                if ans_idx < len(candidates):
                    start = candidates[ans_idx]["plaintext_start_byte"]
                    end = candidates[ans_idx]["plaintext_end_byte"]
                    full_text_bytes = item.get("document_plaintext", "").encode('utf-8')
                    return full_text_bytes[start:end].decode('utf-8')
    except Exception as e:
        logger.warning(f"Could not extract reference answer ({e})")
    return ""


def update_rag_config(config_name, config_data):
    logger.info(f"[{config_name}] Sending new configuration to RAG server...")
    payload = {"rag_settings": config_data}
    try:
        response = requests.patch(CONFIG_API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        logger.info(f"[{config_name}] RAG configuration updated successfully!")
    except requests.exceptions.RequestException as e:
        error_msg = f"Error updating RAG config for {config_name}: {e}"
        logger.error(error_msg)
        send_email_notification(f"FAILED: {config_name}", error_msg)
        raise SystemExit(1)


def get_processed_count(filepath):
    """Counts the number of already processed lines in the output file (for resume functionality)."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


# ==========================================
# MAIN EXECUTION
# ==========================================
def run_all_baselines():
    if not os.path.exists(INPUT_JSONL):
        error_msg = f"'{INPUT_JSONL}' not found."
        logger.error(error_msg)
        send_email_notification("CRITICAL ERROR: Dataset missing", error_msg)
        return

    dataset = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    logger.info(f"Dataset loaded: {len(dataset)} questions found.")

    client = RAGClient(
        base_url=RAG_BASE_URL,
        internal_token=INTERNAL_TOKEN,
        timeout=120
    )

    for config_name, config_data in EXPERIMENTS.items():

        # 1. Filename & timestamp logic (incl. resume check)
        file_prefix = f"CLAPNQ_{config_name}"
        existing_files = glob.glob(os.path.join(OUTPUT_DIR, f"{file_prefix}_*_results.jsonl"))

        if existing_files:
            output_file = sorted(existing_files)[-1]
            base_name = os.path.basename(output_file)
            timestamp = base_name.replace(f"{file_prefix}_", "").replace("_results.jsonl", "")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_{timestamp}_results.jsonl")

        meta_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_{timestamp}_meta.json")

        logger.info("=" * 60)
        logger.info(f"STARTING EXPERIMENT: {config_name} (Run ID: {timestamp})")
        logger.info("=" * 60)

        # 2. Save meta file (if new)
        if not os.path.exists(meta_file):
            meta_data = {
                "run_id": timestamp,
                "experiment_name": config_name,
                "dataset": INPUT_JSONL,
                "rag_settings": config_data
            }
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=4)
            logger.info(f"Metadata saved to: {meta_file}")

        # 3. Update RAG config
        update_rag_config(config_name, config_data)
        time.sleep(3)

        # 4. Check current progress
        processed_count = get_processed_count(output_file)

        if processed_count >= len(dataset):
            logger.info(f"{config_name} is already fully processed ({processed_count} questions). Skipping.")
            continue
        elif processed_count > 0:
            logger.info(f"Checkpoint found: Resuming {config_name} at question {processed_count + 1}...")

        # 5. Send questions to RAG
        for index, item in enumerate(dataset):
            if index < processed_count:
                continue

            question = item.get("question_text", "")
            reference_answer = extract_reference_answer(item)

            if not question:
                continue

            logger.info(f"[{config_name} | {index + 1}/{len(dataset)}] Q: {question[:60]}...")

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

                    with open(output_file, mode="a", encoding="utf-8") as outfile:
                        outfile.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

                    break

                except Exception as e:
                    logger.error(f"Error processing question {index + 1} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(3)
                    else:
                        logger.error(f"Question {index + 1} skipped after {MAX_RETRIES} attempts.")

                finally:
                    if session_id:
                        try:
                            client.delete_thread(session_id)
                        except:
                            pass

            time.sleep(1)

            # Send success email for this config
        success_msg = f"Experiment '{config_name}' has been successfully completed.\nProcessed {len(dataset)} questions."
        send_email_notification(f"SUCCESS: {config_name}", success_msg)

    logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    send_email_notification("ALL DONE", "All baseline experiments have finished successfully.")


if __name__ == "__main__":
    try:
        run_all_baselines()
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Critical script failure:\n{error_trace}")
        send_email_notification("CRITICAL FAILURE", f"The pipeline crashed with the following error:\n\n{error_trace}")
        sys.exit(1)