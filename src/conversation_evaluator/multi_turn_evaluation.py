import json
import os
import gc
from dotenv import load_dotenv, find_dotenv

from ragas.metrics import AspectCritic
from ragas.dataset_schema import MultiTurnSample, EvaluationDataset
from ragas.messages import HumanMessage, AIMessage
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

load_dotenv(find_dotenv(), override=True)


chat_model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=1.0,
    max_retries=10,
    request_timeout=300,
    max_completion_tokens=8192
)

evaluator_llm = LangchainLLMWrapper(chat_model)

_orig_generate = evaluator_llm.generate_text
_orig_agenerate = evaluator_llm.agenerate_text

def _patched_generate(*args, **kwargs):
    kwargs["temperature"] = 1.0
    return _orig_generate(*args, **kwargs)

async def _patched_agenerate(*args, **kwargs):
    kwargs["temperature"] = 1.0
    return await _orig_agenerate(*args, **kwargs)

evaluator_llm.generate_text = _patched_generate
evaluator_llm.agenerate_text = _patched_agenerate

paths = [
    "/home/MH_RAG_DIVE/data/wikipedia/mh_rag_dive/dialogues_20260820_130747.jsonl"
]

for path in paths:
    print(f"\nEvaluating Multi-Turn for: {path}")

    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    list_conversations = []

    for conversation in data:
        conversationlist = conversation['conversation']
        userInput = []

        for turn in conversationlist:
            userInput.append(HumanMessage(content=turn['rag_input']))
            userInput.append(AIMessage(content=turn['rag_answer']))

        sample_conversation = MultiTurnSample(
            user_input=userInput
        )
        list_conversations.append(sample_conversation)

    definition_forgetfulness = "Return 1 if the AI forgets relevant information from earlier in the conversation, showing loss of context or failure to follow up on prior turns; otherwise, return 0."
    forgetfulness_aspect_critic = AspectCritic(
        name="forgetfulness_aspect_critic",
        definition=definition_forgetfulness,
        llm=evaluator_llm,
    )

    definition_contextretention = "Return 1 if the AI clearly retains relevant information from earlier in the conversation, demonstrating strong understanding and continuity across turns; otherwise, return 0."
    context_retention_aspect_critic = AspectCritic(
        name="context_retention_aspect_critic",
        definition=definition_contextretention,
        llm=evaluator_llm,
    )

    print(f"Starting Ragas Evaluation for {len(list_conversations)} conversations...")
    with get_openai_callback() as cb:
        result = evaluate(
            dataset=EvaluationDataset(samples=list_conversations),
            metrics=[forgetfulness_aspect_critic, context_retention_aspect_critic],
            llm=evaluator_llm,
            run_config=RunConfig(max_workers=4)
        )

        token_usage = {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost_usd": cb.total_cost
        }

    result_json = []
    results_df = result.to_pandas()

    for _, row in results_df.iterrows():
        user_input_messages = row["user_input"]
        forgetfulness_critic = row.get("forgetfulness_aspect_critic", 0)
        context_critic = row.get("context_retention_aspect_critic", 0)

        conversation_text = "\n".join([f"{msg['type']}: {msg['content']}" for msg in user_input_messages])

        result_json.append({
            "conversation_log": conversation_text,
            "forgetfulness_aspect_critic": forgetfulness_critic,
            "context_retention_aspect_critic": context_critic
        })

    avg_forgetfulness = sum([entry["forgetfulness_aspect_critic"] for entry in result_json]) / len(result_json)
    avg_context_retention = sum([entry["context_retention_aspect_critic"] for entry in result_json]) / len(result_json)

    final_output = {
        "file_path": path,
        "evaluation_config": {
            "llm_as_a_judge": chat_model.model_name,
        },
        "token_usage_total_run": token_usage,
        "average_forgetfulness_aspect_critic": avg_forgetfulness,
        "average_context_retention_aspect_critic": avg_context_retention,
        "detailed_results": result_json
    }

    print(f"--- RESULTS ---")
    print(f"Forgetfulness: {avg_forgetfulness:.2f} | Context Retention: {avg_context_retention:.2f}")
    print(f"Tokens used: {token_usage['total_tokens']} | Cost: ${token_usage['total_cost_usd']:.4f}")

    final_path = path.replace(".jsonl", "_results_multi_turn.json")

    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    with open(final_path, "w", encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

    gc.collect()