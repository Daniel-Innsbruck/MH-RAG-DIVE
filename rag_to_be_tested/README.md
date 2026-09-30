# RAG Eval API

This directory houses the configurable RAG application used as the target system in our experiments, allowing us to evaluate variations in embedding models, chunk counts, and reasoning strategies.


## Project Structure

```text
├── config.yaml            # Main configuration (RAG settings, prompts)
├── docker-compose.yml     # Docker deployment configuration
├── Dockerfile             # Container definition
├── README.md              # Documentation
├── requirements.txt       # Python dependencies
└── src/                   # Application source code
    ├── config_manager.py  # Handles YAML config parsing
    ├── dtos.py            # Pydantic models for API validation
    ├── logger.py          # Centralized logging setup
    ├── main.py            # FastAPI application entry point
    ├── models/            # LLM API wrappers
    │   ├── chat_gpt.py    # OpenAI integration
    │   └── gemini.py      # Google Gemini integration
    └── qa_chains/         # RAG pipeline implementations
        ├── qa_chain_agentic.py   # Multi-hop/agentic RAG logic
        ├── qa_chain_baseline.py  # Standard/naive RAG logic
        └── shared.py             # Shared utilities (vector store, history)
```

## Features
* **Two RAG Strategies:** 
  * `baseline`: A standard RAG pipeline relying on a single retrieval step, representing the base configuration evaluated in our study.
  * `agentic`: An agentic setup that allows the system to perform multiple retrieval steps instead of relying on a single retrieval step.
* **Model Agnostic:** Wrappers for OpenAI (GPT) and Google (Gemini) APIs.
* **History Ablation:** Configurable conversation history (`all`, `last_n`, `first_n`, `none`).
* Built-in ChromaDB support for various embedding models (e.g., Gemini, BGE-small, All-MiniLM).

---

## Prerequisites
* **Docker** & **Docker Compose** installed on your system.
* API Keys for the LLMs you intend to use (OpenAI and/or Google Gemini).
* Local Chroma Vector Database for Chunk Retrieval (an example on how to generate this is provided in `preprocessing/create_vtarget.py`)

## Setup & Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd <your-repo-folder>/rag_to_be_tested
```

**2. Configure Environment Variables**
Create a `.env` file in `rag_to_be_tested`. You can use the provided `.env.example` as a template:
```bash
cp .env.example .env
```
Open the `.env` file and set your keys:
```ini
# The API key you will use to authenticate requests to this FastAPI app
API_KEY=your_app_api_key_here

# LLM Provider Keys
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**3. Provide a Configuration File**
The application requires a `config.yaml` file in the root directory to define the RAG settings, prompts, and vector store configuration. A template `config.yaml` is provided; update it based on your project's needs. 
*(Note: Ensure this file exists before starting Docker, as it is mounted directly into the container. Alternatively, you can use the `CONFIG_PATH` environment variable to set a custom file path for your config.)*

**4. Start the Application**
Build and start the container using Docker Compose:
```bash
docker-compose up --build -d
```
The API will be available at `http://localhost:8000`.

---

## Usage & API Documentation

Once the container is running, the interactive API documentation (Swagger UI) is automatically available at: **http://localhost:8000/docs**

### Authentication
All main endpoints are protected by an API key. You must include the `X-API-Key` header in your HTTP requests. The value must match the `API_KEY` defined in your `.env` file.

### Core Endpoints Overview
* `GET /health` - Check if the API is running.
* `GET /thread` - Generate a new, unique Thread ID for a conversation session.
* `DELETE /thread/{thread_id}` - Clear a specific conversation's history from memory.
* `GET /config` - Retrieve the current active YAML configuration.
* `PATCH /config` - Partially update the active RAG settings dynamically.
* `POST /rag` - Submit a question to the RAG pipeline. Requires a `thread_id` and the `question` payload.

---

## Local Data & Volumes
The `docker-compose.yml` mounts several local directories to persist data:
* `./logs`: Stores the application logs (`app.log`).
* `./hf_cache`: Caches downloaded HuggingFace embedding models locally.
* `./config.yaml`: Your active configuration file.
* *ChromaDB Directory:* uncomment and configure the ChromaDB volume binding in the `docker-compose.yml`.