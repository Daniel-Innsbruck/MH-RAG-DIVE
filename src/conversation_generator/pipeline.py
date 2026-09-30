import re
import time
import json
import logging
from typing import Dict, Any, Tuple

from common.exceptions import GenerationExhaustedError, DiscoveryFailedError
from common.config import GenConfig
from connectors.rag_client import RAGClient
from conversation_generator.utils.parser import LLMResponseParser
from conversation_validator.conversation_validator import validate_init_prompt_all_in_one, \
    validate_follow_up_question_all_in_one
from context_discoverer.context_discoverer import ContextDiscoverer
import conversation_generator.templates as templates

logger = logging.getLogger(__name__)


def _build_context_string(chunk_list):
    """Helper to concatenate the text snippets of a list of chunks."""
    return "\n\n".join([c['text_snippet'] for c in chunk_list])


def _build_active_context_string_for_validator(active_chunks):
    """Helper to build the formatted context string for the validator."""
    return (
        f"--- CURRENT CONTEXT A ---\n{_build_context_string(active_chunks['A'])}\n\n"
        f"--- CURRENT CONTEXT B ---\n{_build_context_string(active_chunks['B'])}"
    )


# ---------------------------------------------------------
# LLM Retry Loops with separate timing & rejection tracking
# ---------------------------------------------------------

def _generate_and_validate_turn_0(gen_llm, val_llm, chunk_a, chunk_b, t_bridge, role, max_retries=3, val_max_retries=3):
    """
    Generates and validates the initial multi-hop query (Turn 0).

    The Conversation Generator (CG) formulates a question that requires integrating
    information from both context chunks based on the established semantic links.
    The Conversation Validator (CV) then verifies this output[cite: 1].
    """
    parser = LLMResponseParser()
    prompt = templates.CONVERSATION_PROMPTS['init_multihop_prompt'].format(
        Role=role, t_bridge=t_bridge,
        chunk_a=_build_context_string([chunk_a]), chunk_b=_build_context_string([chunk_b])
    )
    combined_docs = f"--- CONTEXT A ---\n{_build_context_string([chunk_a])}\n\n--- CONTEXT B ---\n{_build_context_string([chunk_b])}"

    gen_time_total = 0.0
    val_time_total = 0.0
    rejection_reasons = []

    for attempt in range(max_retries + 1):
        t0 = time.time()
        raw_ans = gen_llm.chat_with_model(prompt)
        gen_time_total += (time.time() - t0)

        parsed_ans = parser.parse_and_validate(raw_ans)
        if parsed_ans:
            logger.info(f"   [Validator] Validating Question (Attempt {attempt + 1})...")

            t1 = time.time()
            validation = validate_init_prompt_all_in_one(parsed_ans, combined_docs, val_llm,
                                                         max_retries=val_max_retries)
            val_time_total += (time.time() - t1)

            if validation and validation.get("correct"):
                return parsed_ans, gen_time_total, val_time_total, rejection_reasons

            reason = validation.get("reason", "Unknown Error")
            prompt_used = validation.get("prompt_used", "N/A")
            rejection_reasons.append({"reason": reason, "validator_prompt": prompt_used})

            logger.warning(f"   [Validator] Question rejected: {reason}. Regenerating...")
            prompt = templates.CONVERSATION_PROMPTS['rephrase_init_prompt'].format(Role=role, reason=reason)

    raise GenerationExhaustedError(f"Turn 0 failed after {max_retries} attempts.")


def _generate_and_validate_follow_up(gen_llm, val_llm, rag_answer, active_chunks, expanding_context, conv_history,
                                     max_retries=3, val_max_retries=3):
    """
    Generates and validates subsequent follow-up queries.

    The Conversation Generator produces follow-up questions based on an expanded context
    while preserving the multi-hop nature of the dialogue.
    """
    parser = LLMResponseParser()

    prompt = templates.CONVERSATION_PROMPTS['follow_up_prompt'].format(
        expanding_context=expanding_context,
        RAG_answer=rag_answer
    )

    current_active_context = _build_active_context_string_for_validator(active_chunks)

    formatted_history = ""
    for turn in conv_history:
        formatted_history += f"User: {turn.get('rag_input')}\nRAG System: {turn.get('rag_answer')}\n\n"

    gen_time_total = 0.0
    val_time_total = 0.0
    rejection_reasons = []

    for attempt in range(max_retries + 1):
        t0 = time.time()
        raw_ans = gen_llm.chat_with_model(prompt)
        gen_time_total += (time.time() - t0)

        parsed_ans = parser.parse_and_validate(raw_ans)

        if parsed_ans:
            logger.info(f"   [Validator] Validating Follow-Up (Attempt {attempt + 1})...")

            t1 = time.time()
            validation = validate_follow_up_question_all_in_one(
                parsed_ans, formatted_history, current_active_context, val_llm, max_retries=val_max_retries
            )
            val_time_total += (time.time() - t1)

            if validation and validation.get("correct"):
                logger.info(f"   [Validator] QA-Pair accepted!")
                parsed_ans["multi_hop_flag"] = 0 # flag not set in follow-up turns
                parsed_ans["logic_type"] = ""

                return parsed_ans, gen_time_total, val_time_total, rejection_reasons

            reason = validation.get("reason", "Unknown Error")
            prompt_used = validation.get("prompt_used", "N/A")
            rejection_reasons.append({"reason": reason, "validator_prompt": prompt_used})

            logger.warning(f"   [Validator] Follow-Up rejected: {reason}. Regenerating...")
            prompt = templates.CONVERSATION_PROMPTS['rephrase_follow_up_prompt'].format(reason=reason)

    raise GenerationExhaustedError(f"Follow-Up failed after {max_retries} attempts.")

def generate_single_conversation(
        config: GenConfig, rag_client: RAGClient, db_connector, cache,
        discoverer_llm, generator_llm_turn_0, generator_llm_followup, validator_llm,
        preloaded_data: dict = None
) -> Tuple[Dict[str, Any], list]:
    """
    Executes a complete synthetic conversation loop.
    """
    generator_llm_turn_0.reset_chat()
    generator_llm_followup.reset_chat()
    validator_llm.reset_chat()
    discoverer_llm.reset_chat()
    if cache: cache.clear_temp_stage()

    thread_id = rag_client.get_thread_id()
    cd = ContextDiscoverer(db_connector=db_connector, llm_model=discoverer_llm, k=4)
    metrics_list = []
    conv_history = []

    try:
        # ---------------------------------------------------------
        # Turn 0: Context Discovery & Generation
        # ---------------------------------------------------------
        root_data = None
        is_new_root = False
        gen_time = 0.0
        val_time = 0.0
        disc_time = 0.0
        turn_0_rejections = []

        if preloaded_data:
            root_data = preloaded_data

        if root_data:
            logger.info(f"[Turn 0] Using existing cache root: {root_data['root_id']}")
            chunk_a, chunk_b = root_data['chunk_a'], root_data['chunk_b']
            question_dict = root_data['initial_question']
            t_bridge = root_data.get('t_bridge', '')
        else:
            logger.info("[Turn 0] Searching for semantic bridge context (Context Discoverer)...")
            t0_disc = time.time()
            discovered = cd.discover_valid_context()
            disc_time = time.time() - t0_disc

            if not discovered:
                raise DiscoveryFailedError("No context found via Discoverer.")

            chunk_a, chunk_b, t_bridge = discovered['chunk_a'], discovered['chunk_b'], discovered['t_bridge']
            logger.info(f"[Turn 0] Context found! Bridging topics: {t_bridge}")

            question_dict, gen_time, val_time, turn_0_rejections = _generate_and_validate_turn_0(
                generator_llm_turn_0, validator_llm, chunk_a, chunk_b, t_bridge, config.generator_role,
                val_max_retries=config.validator_max_retries
            )
            is_new_root = True

        simulated_init_prompt = templates.CONVERSATION_PROMPTS['init_multihop_prompt'].format(
            Role=config.generator_role, t_bridge=t_bridge,
            chunk_a=_build_context_string([chunk_a]), chunk_b=_build_context_string([chunk_b])
        )
        generator_llm_followup.inject_history(simulated_init_prompt, json.dumps(question_dict, ensure_ascii=False))

        active_chunks = {"A": [chunk_a], "B": [chunk_b]}

        if cache:
            if is_new_root and config.build_cache:
                root_id, root_data = cache.prepare_new_root(
                    chunk_a, chunk_b, t_bridge, question_dict, config.alpha, config.beta, config.expansion_dir
                )
            else:
                root_id = root_data['root_id']
            cache.start_new_path(root_id)

        current_question = question_dict
        disc_in, disc_out = discoverer_llm.get_and_reset_turn_tokens()
        gen_in, gen_out = generator_llm_turn_0.get_and_reset_turn_tokens()
        val_in, val_out = validator_llm.get_and_reset_turn_tokens()

        metrics_list.append({
            "turn_index": 0, "is_root_turn": True, "cache_hit": not is_new_root,
            "tokens_discoverer_in": disc_in, "tokens_discoverer_out": disc_out,
            "tokens_generator_in": gen_in, "tokens_generator_out": gen_out,
            "tokens_validator_in": val_in, "tokens_validator_out": val_out,
            "time_discovery_sec": disc_time,
            "time_generation_sec": gen_time,
            "time_validation_sec": val_time,
            "cache_rejection_reason": None,
            "generation_rejection_reasons": turn_0_rejections
        })

        # ---------------------------------------------------------
        # Main Turn Loop (Turn 1 bis N)
        # ---------------------------------------------------------
        for turn_idx in range(config.turns_per_conversation):
            logger.info(f"\n--- Turn {turn_idx} ---")

            logger.info(f"[RAG] Sende Frage: '{current_question.get('rag_input')}'")
            rag_response = rag_client.ask_rag(current_question.get("rag_input"), thread_id)
            rag_answer, rag_context = rag_response.get("answer", ""), rag_response.get("context", "")

            conv_history.append({
                **current_question, "rag_answer": rag_answer, "context": rag_context, "turn_index": turn_idx,
                "ground_truth_chunks": [c['id'] for c in active_chunks["A"]] + [c['id'] for c in active_chunks["B"]]
            })

            if turn_idx >= config.turns_per_conversation - 1:
                break

            # Context Expansion
            next_turn = turn_idx + 1
            expanding_context_parts = []
            if next_turn == config.alpha:
                adj_a = db_connector.get_adjacent_chunk(
                    active_chunks["A"][0]['article_id'], [c['chunk_index'] for c in active_chunks["A"]],
                    active_chunks["A"][0]['total_chunks'], config.expansion_dir
                )
                if adj_a:
                    active_chunks["A"].append(adj_a)
                    expanding_context_parts.append(f"--- Expanded Context A ---\n{adj_a['text_snippet']}")

            if next_turn == config.beta:
                adj_b = db_connector.get_adjacent_chunk(
                    active_chunks["B"][0]['article_id'], [c['chunk_index'] for c in active_chunks["B"]],
                    active_chunks["B"][0]['total_chunks'], config.expansion_dir
                )
                if adj_b:
                    active_chunks["B"].append(adj_b)
                    expanding_context_parts.append(f"--- Expanded Context B ---\n{adj_b['text_snippet']}")

            expanding_context = ""
            if expanding_context_parts:
                expanding_context = "Use this context as additional background for the dialogue:\n\n" + "\n\n".join(
                    expanding_context_parts)

            # Cache & Generate
            while True:
                next_question = None
                cache_accepted = False
                gen_time = 0.0
                val_time = 0.0
                cache_rejection = None
                gen_rejections = []

                if config.use_cache and config.cache_mode == "all" and not is_new_root and cache:
                    cached_bundle, sim_score, _ = cache.find_cache_hit(root_id, turn_idx, rag_answer)
                    if cached_bundle:
                        proposed_question = cached_bundle['next_question_bundle']
                        if config.cache_safeguard:
                            current_active_context = _build_active_context_string_for_validator(active_chunks)
                            history = generator_llm_followup.get_chat_history()

                            t0 = time.time()
                            validation = validate_follow_up_question_all_in_one(proposed_question, history,
                                                                                current_active_context, validator_llm,
                                                                                max_retries=config.validator_max_retries)
                            val_time = time.time() - t0

                            if validation and validation.get('correct'):
                                logger.info("[Cache] Safeguard passed! Cache is injected.")
                                next_question = proposed_question
                                cache_accepted = True
                            else:
                                reason = validation.get('reason',
                                                        'Unknown Error') if validation else 'LLM Validation Error'
                                prompt_used = validation.get('prompt_used', 'N/A') if validation else 'N/A'
                                cache_rejection = {"reason": reason, "validator_prompt": prompt_used}
                                logger.warning(f"[Cache] Safeguard failed: {reason}. Discarding Cache!")
                        else:
                            logger.info("[Cache] Cache used (Safeguard is disabled).")
                            next_question = proposed_question
                            cache_accepted = True

                    if cache_accepted:
                        cache.active_parent_id = cached_bundle['node_id']
                        simulated_prompt = templates.CONVERSATION_PROMPTS['follow_up_prompt'].format(
                            expanding_context=expanding_context, RAG_answer=rag_answer)
                        generator_llm_followup.inject_history(simulated_prompt,
                                                              json.dumps(next_question, ensure_ascii=False))

                if not cache_accepted:
                    logger.info("[Generation] Generating new follow-up question...")

                    next_question, gen_time, val_time, gen_rejections = _generate_and_validate_follow_up(
                        generator_llm_followup, validator_llm, rag_answer, active_chunks, expanding_context, conv_history,
                        val_max_retries=config.validator_max_retries
                    )
                    if cache and config.build_cache and config.cache_mode == "all":
                        cache.stage_tree_node(turn_idx, rag_answer, next_question)

                disc_in, disc_out = discoverer_llm.get_and_reset_turn_tokens()
                gen_in, gen_out = generator_llm_followup.get_and_reset_turn_tokens()
                val_in, val_out = validator_llm.get_and_reset_turn_tokens()

                if not cache_accepted and (gen_in == 0 or gen_out == 0 or val_in == 0 or val_out == 0):
                    logger.warning(
                        f"[Sanity Check] Regeneration faulty (Gen-In:{gen_in}, Out:{gen_out} | Val-In:{val_in}, Out:{val_out}). Retrying turn!"
                    )
                    continue

                if cache_accepted and config.cache_safeguard and (val_in == 0 or val_out == 0):
                    logger.warning(
                        f"[Sanity Check] Cache-Safeguard tokens missing (Val-In:{val_in}, Out:{val_out}). Retrying turn!"
                    )
                    continue

                metrics_list.append({
                    "turn_index": next_turn, "is_root_turn": False, "cache_hit": cache_accepted,
                    "tokens_discoverer_in": disc_in, "tokens_discoverer_out": disc_out,
                    "tokens_generator_in": gen_in, "tokens_generator_out": gen_out,
                    "tokens_validator_in": val_in, "tokens_validator_out": val_out,
                    "time_discovery_sec": 0.0,
                    "time_generation_sec": gen_time,
                    "time_validation_sec": val_time,
                    "cache_rejection_reason": cache_rejection,
                    "generation_rejection_reasons": gen_rejections
                })
                break

            current_question = next_question

        if cache and config.build_cache and config.cache_mode == "all":
            if is_new_root:
                cache.save_root(root_data)
            cache.commit_tree(root_id)

        parent_a_label = f"{chunk_a.get('article_title', 'Unknown')}_{chunk_a['article_id']}"
        parent_b_label = f"{chunk_b.get('article_title', 'Unknown')}_{chunk_b['article_id']}"

        return {
            "parent_doc_A": parent_a_label,
            "parent_doc_B": parent_b_label,
            "role": config.generator_role,
            "conversation": conv_history
        }, metrics_list

    except (GenerationExhaustedError, DiscoveryFailedError) as e:
        e.partial_metrics = metrics_list
        raise e
    finally:
        if 'thread_id' in locals() and thread_id:
            try:
                rag_client.delete_thread(thread_id)
                logger.info(f"RAG thread cleaned up: {thread_id}")
            except Exception as delete_err:
                logger.warning(f"Could not delete RAG thread {thread_id}: {delete_err}")