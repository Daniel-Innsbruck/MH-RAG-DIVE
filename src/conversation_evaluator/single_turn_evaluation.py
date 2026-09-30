import os
import sys
import json
import smtplib
import traceback
import gc
import logging
from pathlib import Path
from email.mime.text import MIMEText

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from ragas.metrics import AspectCritic
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, context_precision, context_recall

# ==========================================
# Config & IMPORTS
# ==========================================
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from common.config import GenConfig
from common.retry import retry_with_backoff

load_dotenv(find_dotenv(), override=True)
config = GenConfig()

paths = [
    "/home/MH_RAG_DIVE/data/wikipedia/mh_rag_dive/dialogues_20260820_130747.jsonl"
]

token_tracker = OpenAICallbackHandler()

def safe_val(val):
    import math
    if val is None: return None
    if isinstance(val, float) and math.isnan(val): return None
    return float(val)

# ==========================================
# LLM & EMBEDDINGS SETUP
# ==========================================
chat_model = ChatOpenAI(
    model=config.evaluator_model,
    temperature=1.0,
    max_completion_tokens=16384, # high out tokens for less ragas out of token limit errors
    request_timeout=240,
    callbacks=[token_tracker]
)
evaluator_llm = LangchainLLMWrapper(chat_model)

_orig_generate = evaluator_llm.generate_text
_orig_agenerate = evaluator_llm.agenerate_text


# Patch of Ragas method: evaluator-model is a reasoning-model an needs temperature=1
@retry_with_backoff(Exception, max_retries=6, backoff_base=2.0)
def _patched_generate(*args, **kwargs):
    kwargs["temperature"] = 1.0
    return _orig_generate(*args, **kwargs)


@retry_with_backoff(Exception, max_retries=6, backoff_base=2.0)
async def _patched_agenerate(*args, **kwargs):
    kwargs["temperature"] = 1.0
    return await _orig_agenerate(*args, **kwargs)


evaluator_llm.generate_text = _patched_generate
evaluator_llm.agenerate_text = _patched_agenerate

embed_model = OpenAIEmbeddings(model="text-embedding-3-small", callbacks=[token_tracker])
evaluator_embeddings = LangchainEmbeddingsWrapper(embed_model)

definition_correctness = "Return 1 if the AI answers the question correct; otherwise, return 0."
correctness_aspect_critic = AspectCritic(
    name="correctness_aspect_critic",
    definition=definition_correctness,
    llm=evaluator_llm,
)
metrics_to_use = [correctness_aspect_critic, faithfulness, context_precision, context_recall]


def calculate_safe_average(result_list, key):
    valid_entries = [entry[key] for entry in result_list if
                     key in entry and entry[key] == entry[key] and entry[key] is not None]
    if not valid_entries:
        return 0.0
    return sum(valid_entries) / len(valid_entries)


# ==========================================
# EVALUATION CORE
# ==========================================
def run_evaluation_core(samples_list, checkpoint_file):
    processed_inputs = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    processed_inputs.add(json.loads(line).get("content", ""))

    chunk_size = 5

    for i in range(0, len(samples_list), chunk_size):
        chunk = samples_list[i:i + chunk_size]
        chunk_to_process = [s for s in chunk if s.user_input not in processed_inputs]

        if not chunk_to_process:
            continue

        print(f"Processing Batch {i // chunk_size + 1}...")

        try:
            result = evaluate(
                dataset=EvaluationDataset(samples=chunk_to_process),
                metrics=metrics_to_use,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                raise_exceptions=False
            )
            with open(checkpoint_file, 'a', encoding='utf-8') as f:
                for _, row in result.to_pandas().iterrows():
                    f_val = safe_val(row.get("faithfulness"))
                    if f_val is None:
                        print(
                            f"WARNING: Ragas couldn't compute Faithfulness for '{row.get('user_input', '')[:40]}...'!")

                    record = {
                        "content": row.get("user_input", ""),
                        "correctness_aspect_critic": safe_val(row.get("correctness_aspect_critic")),
                        "faithfulness_aspect_critic": f_val,
                        "context_precision_critic": safe_val(row.get("context_precision")),
                        "context_recall_critic": safe_val(row.get("context_recall"))
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as e:
            error_msg = f"Fehler bei Batch {i // chunk_size + 1}: {e}"
            print(error_msg)

    total_tokens = token_tracker.total_tokens
    prompt_tokens = token_tracker.prompt_tokens
    completion_tokens = token_tracker.completion_tokens
    total_cost_usd = token_tracker.total_cost

    final_loaded_results = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            final_loaded_results = [json.loads(line) for line in f if line.strip()]

    average_scores = {
        "correctness_aspect_critic": calculate_safe_average(final_loaded_results, "correctness_aspect_critic"),
        "faithfulness_aspect_critic": calculate_safe_average(final_loaded_results, "faithfulness_aspect_critic"),
        "context_precision_critic": calculate_safe_average(final_loaded_results, "context_precision_critic"),
        "context_recall_critic": calculate_safe_average(final_loaded_results, "context_recall_critic")
    }

    token_usage = {
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_cost_usd": total_cost_usd
    }

    return final_loaded_results, average_scores, token_usage


# ==========================================
# MAIN LOOP
# ==========================================
def main():
    try:
        for path in paths:
            print(f"\nEvaluating: {path}")

            data = []
            with open(path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]

            list_all_conversations = [
                SingleTurnSample(
                    user_input=turn.get('question', ''),
                    reference=turn.get('answer', ''),
                    response=turn.get('rag_answer', ''),
                    retrieved_contexts=turn.get('context', []),
                )
                for conversation in data
                for turn in conversation.get('conversation', [])
            ]

            checkpoint_file = path.replace('.jsonl', '_ragas_checkpoint.jsonl')
            overall_results, overall_averages, overall_tokens = run_evaluation_core(list_all_conversations,
                                                                                    checkpoint_file)

            final_output = {
                "file_path": path,
                "evaluation_config": {
                    "llm_as_a_judge": config.evaluator_model,
                    "embedding_model": embed_model.model
                },
                "overall_evaluation": {
                    "average_scores": overall_averages,
                    "token_usage": overall_tokens,
                    "detailed_results": overall_results
                }
            }

        final_path = path.replace('.jsonl', '_evaluation_results.json')
        with open(final_path, "w", encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        print(f"Gespeichert unter {final_path}")

        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"Clearing temporary files: {os.path.basename(checkpoint_file)}")

        token_tracker.total_tokens = 0
        token_tracker.prompt_tokens = 0
        token_tracker.completion_tokens = 0
        token_tracker.total_cost = 0.0
        token_tracker.successful_requests = 0
        gc.collect()

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Critical Error:\n{error_trace}")


if __name__ == '__main__':
    main()