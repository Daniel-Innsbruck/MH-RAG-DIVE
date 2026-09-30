from pydantic_settings import BaseSettings
from pathlib import Path

SENSITIVE_FIELDS = {"rag_api_key", "openai_api_key", "google_api_key"}

class GenConfig(BaseSettings):
    """
    Central configuration for the Synthetic Dialogue Generation Pipeline.

    This class manages all environment variables and default settings required
    to run the MH-RAG-DIVE evaluation, including target system URLs, LLM
    model selection, API keys, database paths, and experimental hyperparameters.
    """
    # --- RAG Target ---
    rag_base_url: str = "localhost:8000/"
    rag_api_key: str = ""

    # --- Industry Use Case ---
    k5_service_name: str = "aio"
    k5_tenant_name: str = "evalrag"

    # --- LLM Models ---
    discoverer_model: str = "gemini-3.1-flash-lite"
    generator_model_turn_0: str = "gemini-3.1-pro-preview"
    generator_model_followup: str = "gemini-3.1-flash-lite"
    validator_model: str = "gpt-5.6-luna"
    evaluator_model: str = "gpt-5-nano"

    # --- API Keys ---
    openai_api_key: str = ""
    google_api_key: str = ""

    # --- Database (ChromaDB) ---
    chroma_db_path: str = "enter_path_to_your_veval_chromadb_here"
    chroma_collection_name: str = "enter_veval_collection_name_here"

    # --- MH_RAG_DIVE Hyperparameters ---
    turns_per_conversation: int = 5
    max_conversations: int = 100
    alpha: int = 2
    beta: int = 3
    expansion_dir: str = "below"
    generator_role: str = "Your questions are very short and precise"
    validator_max_retries: int = 5 # in case validation format is bad

    # --- Caching ---
    build_cache: bool = True
    use_cache: bool = True
    cache_mode: str = "all"
    cache_tau: float = 0.1
    cache_safeguard: bool = True

    # --- Output ---
    output_dir: Path = Path("data")
    output_file_name: str = "generated_dialogues.jsonl"

    model_config = {
        "env_prefix": "GEN_",
        "env_file": ".env",
        "extra": "ignore",
    }

    def override(self, **kwargs) -> "GenConfig":
        """
        Creates and returns a new configuration instance by overriding specific fields.

        Args:
            **kwargs: Key-value pairs of the configuration settings to update.

        Returns:
            GenConfig: A new instance of GenConfig with the updated values.
        """
        return GenConfig(**{**self.model_dump(), **kwargs})

    def get_safe_dump(self) -> dict:
        """
        Dumps the configuration to a dictionary while stripping all sensitive API keys.

        This is useful for safely logging the configuration or saving it to a metadata
        file alongside the experiment results without leaking credentials.

        Returns:
            dict: The configuration dictionary safe for logging.
        """
        dump = self.model_dump()
        for field in SENSITIVE_FIELDS:
            dump.pop(field, None)
        dump["output_dir"] = str(dump["output_dir"])
        return dump