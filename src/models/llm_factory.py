from models.chat_gpt import ChatGPT
from models.gemini import GEMINI

def get_llm(model_name: str):
    """
    Automatically instantiates the correct LLM wrapper based on the provided model name.

    Args:
        model_name (str): The identifier of the LLM model to load.

    Returns:
        Union[ChatGPT, GEMINI]: The instantiated model wrapper.

    Raises:
        ValueError: If the model prefix is not recognized.
    """
    model_name_lower = model_name.lower()
    if "gpt" in model_name_lower or "o3" in model_name_lower or "o3" in model_name_lower or "o4" in model_name_lower:
        return ChatGPT(model_name)
    elif "gemini" in model_name_lower:
        return GEMINI(model_name)
    else:
        raise ValueError(f"Unknown model prefix for: {model_name}. Supported families are 'gpt/o-series' and 'gemini'.")