from pydantic import BaseModel, Field
from enum import Enum

class HistoryMode(str, Enum):
    """Enumeration of supported chat history retention strategies."""
    ALL = "all"
    LAST_N = "last_n"
    FIRST_N = "first_n"
    FIRST_ONLY = "first_only"
    NONE = "none"

class RagSettingsDTO(BaseModel):
    """Data transfer object for updating RAG pipeline configuration parameters."""
    k_chunks: int | None = Field(default=None, ge=1)
    chroma_collection_name: str | None = Field(default=None)
    chroma_directory: str | None = Field(default=None)
    llm_model: str | None = Field(default=None)
    embedding_model: str | None = Field(default=None)
    system_prompt: str | None = Field(default=None)
    history_mode: HistoryMode | None = Field(default=None)
    history_n: int | None = Field(default=None, ge=0)
    strategy: str | None = Field(default=None, description="'baseline' or 'agentic'")
    max_hops: int | None = Field(default=None, ge=1)
    agentic_system_prompt: str | None = Field(default=None)
    agentic_user_prompt: str | None = Field(default=None)

class ConfigRequest(BaseModel):
    """Payload schema for reading or updating the application configuration."""
    rag_settings: RagSettingsDTO | None = Field(default=None)

class QuestionRequest(BaseModel):
    """Payload schema for submitting a user question to the RAG pipeline."""
    question: str
    thread_id: str

class HealthResponse(BaseModel):
    """Response schema for the API health check endpoint."""
    status: str

class ThreadResponse(BaseModel):
    """Response schema providing a newly generated conversation thread ID."""
    thread_id: str

class RagResponse(BaseModel):
    """Response schema containing the generated answer and the retrieved context."""
    answer: str
    context: list[str]

class ConfigUpdateResponse(BaseModel):
    """Response schema confirming a successful configuration update."""
    status: str
    config: ConfigRequest