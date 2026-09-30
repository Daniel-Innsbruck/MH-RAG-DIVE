import time
from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings

from src.logger import logger
from src.models.chat_gpt import ChatGPT
from src.models.gemini import GEMINI

conversation_store: Dict[str, List[BaseMessage]] = {}

# Caching für ChromaDB
_current_collection = None
_current_emb_model = None
_vector_store = None


def get_vector_store(config: dict):
    """
    Initializes and caches a Chroma vector store based on the configuration.

    Args:
        config (dict): The configuration dictionary containing Chroma settings.

    Returns:
        Chroma: The initialized Chroma vector store instance.
    """
    global _current_collection, _current_emb_model, _vector_store
    col = config["chroma_collection_name"]
    emb = config["embedding_model"]
    dir = config.get("chroma_directory", "/app/data/chroma_db")

    if _vector_store is None or _current_collection != col or _current_emb_model != emb:
        logger.info(f"Initializing VectorStore: {col}, Embeddings: {emb}, Directory: {dir}")
        emb_lower = emb.lower()

        if "gemini" in emb_lower:
            embeddings = GoogleGenerativeAIEmbeddings(model=emb)
        elif "ollama" in emb_lower or "nomic" in emb_lower:
            clean_name = emb.split("/")[-1] if "/" in emb else emb
            embeddings = OllamaEmbeddings(model=clean_name)
        elif any(hf_tag in emb_lower for hf_tag in ["baai", "sentence-transformers", "minilm", "bge"]):
            logger.info(f"Loading local HuggingFace Embedding: {emb}")
            embeddings = HuggingFaceEmbeddings(model_name=emb)
        else:
            raise ValueError(f"Unsupported embedding model: '{emb}'")

        _vector_store = Chroma(
            collection_name=col,
            persist_directory=dir,
            embedding_function=embeddings
        )
        _current_collection = col
        _current_emb_model = emb

    return _vector_store


def get_filtered_history(thread_id: str, config: dict) -> List[BaseMessage]:
    """
    Retrieves and filters the chat history for a specific thread based on config rules.

    Args:
        thread_id (str): The unique identifier for the conversation thread.
        config (dict): The configuration dictionary. Relevant keys include:
            - 'history_mode' (str): Determines how history is managed. Options:
                - 'all': Includes the entire conversation history.
                - 'last_n': Keeps only the last 'n' conversation rounds.
                - 'first_n': Uses only the first 'n' conversation rounds.
                - 'none': Disables memory (returns an empty history).
            - 'history_n' (int): The number of conversation rounds (question + answer)
              to keep when mode is 'last_n' or 'first_n'.

    Returns:
        List[BaseMessage]: A filtered list of conversation messages.
    """
    history = conversation_store.get(thread_id, [])
    mode = config["history_mode"]

    if mode == "none":
        return []
    elif mode == "all":
        return history
    elif mode == "first_n":
        n = config["history_n"]
        return history[:n * 2]
    elif mode == "last_n":
        n = config["history_n"]
        msgs_to_keep = n * 2
        return history[-msgs_to_keep:] if msgs_to_keep > 0 else []
    else:
        raise ValueError(f"Invalid 'history_mode': '{mode}'.")


def format_history_to_string(history: List[BaseMessage]) -> str:
    """
    Formats a list of BaseMessage objects into a readable text format.

    Args:
        history (List[BaseMessage]): The list of messages to format.

    Returns:
        str: A formatted string representation of the chat history.
    """
    if not history:
        return "No previous conversation."
    return "\n".join(f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in history)


def llm_invoke_with_retry(llm_connector, prompt_text: str, max_retries=3, wait_time=60):
    """
    Invokes the LLM with a retry mechanism for rate limits (429) and server errors (503).

    Args:
        llm_connector: The LLM client wrapper instance.
        prompt_text (str): The prompt to send to the LLM.
        max_retries (int): Maximum number of retry attempts. Default is 3.
        wait_time (int): Seconds to wait before retrying. Default is 60.

    Returns:
        str: The LLM's response, or None if it fails.

    Raises:
        Exception: If the call fails for reasons other than rate limits,
                   or if max retries are exceeded.
    """
    for attempt in range(max_retries):
        try:
            return llm_connector.prompt(prompt_text)
        except Exception as e:
            if '429' in str(e) or '503' in str(e) or 'RateLimit' in str(e):
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit / Service unavailable. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            logger.error(f"LLM invoke failed: {str(e)}", exc_info=True)
            raise e
    return None


def get_llm_connector(model_name: str):
    """
    Factory function to instantiate the correct LLM connector based on the model name.

    Args:
        model_name (str): The name of the LLM model to use (e.g., 'gpt-5-nano', 'gemini-3.1-flash-lite').

    Returns:
        Union[ChatGPT, GEMINI]: An instantiated wrapper for the requested model.

    Raises:
        ValueError: If the model_name is not supported.
    """
    name_lower = model_name.lower()
    if "gpt" in name_lower:
        logger.info(f"Using ChatGPT connector for model: {model_name}")
        return ChatGPT(model_name)
    elif "gemma" in name_lower or "gemini" in name_lower:
        logger.info(f"Using Google/Gemini API connector for model: {model_name}")
        return GEMINI(model_name)
    else:
        raise ValueError(f"Unknown LLM type: {model_name}")