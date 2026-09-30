import sys
import time
import json
import logging
from pathlib import Path
import typer
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from common.config import GenConfig
from common.exceptions import GenerationExhaustedError, DiscoveryFailedError
from connectors.rag_client import RAGClient
# from connectors.k5_client import K5RAGClient
from models.llm_factory import get_llm
from connectors.chroma_connector import ChromaConnector
from conversation_cacher.conversation_cacher import ConversationCacher
from conversation_generator.pipeline import generate_single_conversation

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(help="MH RAG DIVE - Synthetic Dialogue Generator")

#def save_run_metadata(config: GenConfig, rag_client: K5RAGClient, run_id: str):
def save_run_metadata(config: GenConfig, rag_client: RAGClient, run_id: str):
    try:
        rag_config = rag_client.get_config()
    except Exception as e:
        logger.warning(f"Could not fetch RAG config: {e}")
        rag_config = {"error": "Could not fetch RAG config"}

    run_meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "pipeline_config": config.get_safe_dump(),
        "rag_container_config": rag_config
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = config.output_dir / f"run_meta_{run_id}.json"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=4, ensure_ascii=False)
    logger.info(f"Run metadata saved: {meta_path}")


@app.command()
def generate(
        conversations: int = typer.Option(None, help="Number of dialogues to generate (overrides .env)"),
        mode: str = typer.Option("fill", help="Mode: 'fill' (generate new), 'reuse' (read only), 'hybrid' (both)"),
        # --- Type-safe RAG parameters for runtime override ---
        k_chunks: int = typer.Option(None, help="Number of chunks to retrieve"),
        chroma_collection_name: str = typer.Option(None, help="Name of the ChromaDB collection"),
        llm_model: str = typer.Option(None, help="LLM model name"),
        embedding_model: str = typer.Option(None, help="Embedding model name"),
        strategy: str = typer.Option(None, help="RAG strategy: 'baseline' or 'agentic'"),
        history_mode: str = typer.Option(None, help="History mode: all, last_n, first_n, none"),
        history_n: int = typer.Option(None, help="Number of turns for history_mode"),
        max_hops: int = typer.Option(None, help="Maximum agent hops"),
):
    overrides = {}
    if conversations:
        overrides["max_conversations"] = conversations

    if mode == "fill":
        logger.info("MODE: FILL -> Ignoring cache for reading, building new cache.")
        overrides["use_cache"] = False
        overrides["build_cache"] = True
    elif mode == "reuse":
        logger.info("MODE: REUSE -> Using existing cache only, not building new branches.")
        overrides["use_cache"] = True
        overrides["build_cache"] = False
    elif mode == "hybrid":
        logger.info("MODE: HYBRID -> Loading from cache, generating and saving on cache miss.")
        overrides["use_cache"] = True
        overrides["build_cache"] = True
    else:
        logger.warning(f"Unknown mode '{mode}'. Falling back to 'fill'.")
        overrides["use_cache"] = False
        overrides["build_cache"] = True

    config = GenConfig().override(**overrides) if overrides else GenConfig()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = config.output_dir / f"pipeline_log_{run_id}.txt"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Log file created for this run: {log_file}")

    rag_client = RAGClient(config.rag_base_url, config.rag_api_key)
    # rag_client = K5RAGClient(
    #     base_url=config.rag_base_url,
    #     internal_token=config.rag_api_key,
    #     service_name=config.k5_service_name,
    #     tenant_name=config.k5_tenant_name
    # )
    rag_payload = {}
    if k_chunks is not None: rag_payload["k_chunks"] = k_chunks
    if chroma_collection_name: rag_payload["chroma_collection_name"] = chroma_collection_name
    if llm_model: rag_payload["llm_model"] = llm_model
    if embedding_model: rag_payload["embedding_model"] = embedding_model
    if strategy: rag_payload["strategy"] = strategy
    if history_mode: rag_payload["history_mode"] = history_mode
    if history_n is not None: rag_payload["history_n"] = history_n
    if max_hops is not None: rag_payload["max_hops"] = max_hops

    if rag_payload:
        logger.info(f"Updating RAG server configuration with parameters: {rag_payload}")
        try:
            rag_client.update_config(rag_payload)
        except Exception as e:
            logger.error(f"Error updating RAG config: {e}")
            raise typer.Exit(code=1)

    save_run_metadata(config, rag_client, run_id)
    save_run_metadata(config, rag_client, run_id)

    logger.info("Initializing modules...")
    discoverer_llm = get_llm(config.discoverer_model)
    generator_llm_turn_0 = get_llm(config.generator_model_turn_0)
    generator_llm_followup = get_llm(config.generator_model_followup)
    validator_llm = get_llm(config.validator_model)

    db_connector = ChromaConnector(config.chroma_db_path, collection_name=config.chroma_collection_name)
    cache = ConversationCacher(tau=config.cache_tau, safeguard=config.cache_safeguard) if (
                config.use_cache or config.build_cache) else None

    out_file = config.output_dir / f"dialogues_{run_id}.jsonl"
    metrics_file = config.output_dir / f"metrics_{run_id}.jsonl"

    if mode in ["hybrid", "reuse"] and cache:
        target_filenames = cache.get_all_root_filenames()
        if target_filenames:
            # One dialogue per root
            target_count = len(target_filenames)
            logger.info(
                f"Mode '{mode}': Found {target_count} root files. Target: {target_count} successful dialogues.")
        else:
            target_filenames = []
            target_count = config.max_conversations
            logger.warning(f"Mode '{mode}' selected, but no cache files found. Switching to 'fill' behavior.")
    else:
        # Fill mode
        target_filenames = []
        target_count = config.max_conversations
        logger.info(f"Mode '{mode}': Generating {target_count} completely new dialogues (Turn 0 Discovery).")

    consecutive_fails = 0
    success_count = 0
    attempt_index = 0

    while success_count < target_count:
        current_conv_id = success_count + 1
        attempt_index += 1

        if consecutive_fails >= 15:
            error_msg = f"Pipeline {run_id} aborted! 15 critical errors in a row."
            logger.error(error_msg)
            raise typer.Exit(code=1)

        logger.info(f"\n=======================================================")
        logger.info(
            f"   STARTING DIALOGUE {current_conv_id}/{target_count} (Total attempt #{attempt_index} | Consecutive fails: {consecutive_fails})")
        logger.info(f"=======================================================")

        filename = None
        if target_filenames and success_count < len(target_filenames):
            filename = target_filenames[success_count]

        current_root = None
        if filename and cache:
            current_root = cache.get_root_by_filename(filename)
            if current_root:
                logger.info(f"Loading root from file: {filename}")
            else:
                logger.warning(f"Could not load file {filename}. Generating new root (Turn 0 Discovery)...")
        try:
            conv_data, metrics = generate_single_conversation(
                config, rag_client, db_connector, cache,
                discoverer_llm, generator_llm_turn_0, generator_llm_followup, validator_llm, preloaded_data=current_root
            )

            thread_id_to_clean = conv_data.get("thread_id")
            if thread_id_to_clean:
                rag_client.delete_thread(thread_id_to_clean)

            with open(out_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(conv_data, ensure_ascii=False) + "\n")

            with open(metrics_file, 'a', encoding='utf-8') as f:
                for m in metrics:
                    m["conversation_id"] = current_conv_id
                    m["run_id"] = run_id
                    m["status"] = "success"
                    m["root_id"] = filename.replace(".json", "") if filename else "NEW"
                    f.write(json.dumps(m) + "\n")

            success_count += 1
            consecutive_fails = 0
            logger.info(f"Dialogue {current_conv_id} successfully generated and saved.")
            logger.info(f"Cooldown: Waiting 60s before the next dialogue...")
            time.sleep(60)

        except (GenerationExhaustedError, DiscoveryFailedError) as e:
            logger.warning(f"Dialogue {current_conv_id} aborted (limit reached): {e}")
            consecutive_fails += 1

            if hasattr(e, 'partial_metrics') and e.partial_metrics:
                with open(metrics_file, 'a', encoding='utf-8') as f:
                    for m in e.partial_metrics:
                        m["conversation_id"] = current_conv_id
                        m["run_id"] = run_id
                        m["status"] = "failed_and_discarded"
                        m["root_id"] = filename.replace(".json", "") if filename else "NEW"
                        f.write(json.dumps(m) + "\n")
            logger.info("Cooldown after failure: Waiting 60s...")
            time.sleep(60)

        except Exception as e:
            error_msg_lower = str(e).lower()
            if any(fatal in error_msg_lower for fatal in
                   ["401", "403", "404", "model_not_found", "invalid_request_error"]):
                fatal_msg = f"Pipeline {run_id} terminated due to FATAL API error: {e}"
                logger.error(fatal_msg)
                raise typer.Exit(code=1)

            logger.exception(f"Unexpected crash during dialogue {current_conv_id}: {e}")
            consecutive_fails += 1

            logger.info("Cooldown after exception: Waiting 120s..")
            time.sleep(120)

    success_msg = f"Pipeline {run_id} finished successfully!\nGenerated dialogues: {success_count}/{target_count}\nMode: {mode}"
    logger.info(f"\n{success_msg}")

if __name__ == "__main__":
    app()