from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
import uuid
import os

from src.config_manager import load_config, save_config
from src.qa_chains.qa_chain_baseline import baseline_rag
from src.qa_chains.qa_chain_agentic import agentic_rag
from src.qa_chains.shared import conversation_store
from src.logger import logger
from src.dtos import *
app = FastAPI(title="RAG Eval API")

# --- API-Key Protection ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validates the API key from the request header."""
    expected_api_key = os.getenv("API_KEY")
    if not expected_api_key:
        logger.error("API_KEY is not set in current environment!")
        raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY not set")

    if api_key_header == expected_api_key:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# Health-Check
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Performs a basic health check to ensure the API is running."""
    return {"status": "healthy"}

@app.get("/thread", dependencies=[Depends(get_api_key)], response_model=ThreadResponse)
def get_thread_id():
    """Generates and returns a new unique thread ID for a conversation session."""
    return {"thread_id": str(uuid.uuid4())}

@app.delete("/thread/{thread_id}", dependencies=[Depends(get_api_key)])
def delete_thread(thread_id: str):
    """Deletes a specific conversation thread and its history from memory."""
    try:
        if thread_id in conversation_store:
            del conversation_store[thread_id]
            logger.info(f"Thread {thread_id} deleted from memory.")
            return {"status": "success", "message": f"Thread {thread_id} deleted"}
        else:
            return {"status": "success", "message": f"Thread {thread_id} not found"}
    except Exception as e:
        logger.error(f"Error deleting thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/config", dependencies=[Depends(get_api_key)], response_model=ConfigRequest)
def get_config():
    """Retrieves the current application configuration."""
    try:
        return load_config()
    except Exception as e:
        logger.error("Error reading configuration", exc_info=True)
        raise HTTPException(status_code=500, detail="Config read error")


@app.patch("/config", dependencies=[Depends(get_api_key)], response_model=ConfigUpdateResponse)
def patch_config(new_config: ConfigRequest):
    """Partially updates the application configuration with the provided data."""
    try:
        current_config = load_config() or {}

        update_data = new_config.model_dump(exclude_unset=True, exclude_none=True)

        if "rag_settings" in update_data and isinstance(update_data["rag_settings"], dict):
            if "rag_settings" not in current_config or current_config["rag_settings"] is None:
                current_config["rag_settings"] = {}

            for key, value in update_data["rag_settings"].items():
                if value is not None:
                    current_config["rag_settings"][key] = value

        save_config(current_config)
        logger.info(f"Configuration partially updated: {update_data}")
        return {"status": "success", "config": current_config}

    except Exception as e:
        logger.error(f"Error saving configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not save config")

@app.post("/rag", dependencies=[Depends(get_api_key)], response_model=RagResponse)
def ask_question(request: QuestionRequest):
    """Processes a user question using the configured RAG strategy."""
    try:
        logger.debug(f"New RAG request for thread: {request.thread_id} | Question: {request.question}")

        config = load_config()["rag_settings"]
        strategy = config["strategy"]

        if strategy == "agentic":
            result = agentic_rag(request.question, request.thread_id)
        elif strategy == "baseline":
            result = baseline_rag(request.question, request.thread_id)
        else:
            logger.warning(f"Invalid or missing RAG strategy requested: '{strategy}'")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy: '{strategy}'. Allowed values are 'baseline' or 'agentic'."
            )

        return {"answer": result["answer"], "context": result["context"]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error during RAG request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error. See logs.")