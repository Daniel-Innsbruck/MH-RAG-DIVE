# MH-RAG-DIVE

**Dynamic and Cost-Efficient Multi-Hop RAG Evaluation via Tree-Cached Dialogue Simulation**

This repository contains the source code, experimental data, and evaluation scripts for the paper introducing **MH-RAG-DIVE**.

By leveraging granular document chunking, incremental context expansion, and a dynamic Semantic Conversation Cache, MH-RAG-DIVE simulates realistic, multi-hop user interactions while reducing generation token costs by over 70% compared to full-document baselines (RAG-DIVE).

---

## Repository Overview

The repository is structured to reflect the theoretical architecture outlined in the paper, separating context discovery, generation, validation, and caching mechanisms.

### Folder Structure

| Folder                  | Description                                                                                                                                                                                                                                                                                       |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`src/`**              | The core codebase of the MH-RAG-DIVE framework. Contains the modular components for context discovery, generation, validation, evaluation, caching, and model wrappers. The entry point is `cli.py`.                                                                                              |
| **`analysis/`**         | Contains a jupyter notebook used to analyze the caching efficacy and the token consumption of the framework against the RAG-DIVE baseline.                                                                                                                                                        |
| **`preprocessing/`**    | Contains xamplary scripts for creating $V_{eval}$ and $V_{target}$ from a corpus of Wikipedia Articles                                                                                                                                                                                            |
| **`data/`**             | Contains the experimental outputs. Includes the `dialogue_cache/` (storing the semantic dialogue trees), generated conversations (`.jsonl`), runtime metrics (`metrics_*.jsonl`), and configuration snapshots (`run_meta.json`). Segmented by dataset (e.g., `wikipedia/`, `industry_use_case/`). |
| **`human_evaluation/`** | Contains Excel sheets and raw samples used for the manual human validation of the synthetically generated multi-hop dialogues.                                                                                                                                                                    |
| **`rag_to_be_tested/`** | A standalone, containerized RAG system used as the evaluation target during the experiments with the Wikipedia corpus.                                                                                                                                                                            |

---

## System Architecture

The framework consists of five core components that operate in a continuous loop to simulate a user interacting with a RAG system over $N=5$ turns:

1. **Context Discoverer (CD):** Executes a two-stage process (filtered semantic search + LLM validation) across the evaluation vector database ($V_{eval}$) to find semantically linked document chunks ($d_A$ and $d_B$) and establish a thematic bridge ($T_{bridge}$) for multi-hop reasoning.
2. **Conversation Generator (CG):** Assumes a specific persona to generate the initial multi-hop query (Turn 0) and subsequent follow-up questions based on incrementally expanding contexts ($\alpha, \beta$).
3. **Conversation Validator (CV):** Acts as a strict gatekeeper, verifying query alignment, multi-hop integrity, and logical flow. Rejected queries trigger a targeted regeneration loop.
4. **Conversation Cacher (CC):** Dynamically maps conversational trajectories into a tree structure. By computing the cosine similarity of RAG responses against the cached follow-up bundles of that conversation, it reuses previously validated follow-up questions to bypass LLM generations of the CG.
5. **Conversation Evaluator (CE):** (Post-generation) Evaluates the target RAG system's responses using established metrics (e.g., RAGAS) across both single-turn and multi-turn dimensions.

---

## Preparation & Evaluation Database ($V_{eval}$)

To evaluate a target system, MH-RAG-DIVE requires an independent evaluation vector database ($V_{eval}$) populated with correctly formatted and indexed document chunks. 

This repository includes an exemplary preprocessing pipeline (documented in the Jupyter Notebook in  `preprocessing/`), demonstrating how $V_{eval}$ can be constructed using the Wikipedia dataset:
1. **Data Acquisition:** Sourced from Wikipedia articles linked via historical revision IDs (`oldid`) from the ClapNQ dataset to ensure exact version consistency.
2. **Data Cleaning:** Raw wikitext is parsed using `mwparserfromhell` into flat section-based JSON structures. Administrative and noise sections (such as *references, footnotes, external links, and see also*) are automatically filtered out via a strict blacklist.
3. **Chunking & Metadata Enrichment:** Text is split recursively (chunk size: 3,500 characters, overlap: 525 characters) to match ~1,000 token segments. Each chunk is retroactively enriched with descriptive metadata (article titles, section names) and structural markers (parent ID $P(d_i)$, position index $Idx(d_i)$, and total chunks $N(d_i)$) to support deterministic context expansion.

---

## Getting Started

### Prerequisites
- **Docker** and **Docker Compose**
- **Python 3.12+** and **Poetry** (for local evaluation scripts)
- API Keys for the underlying LLMs and Embeddings (OpenAI, Google Gemini)
- Running Test RAG System (see `rag_to_be_tested/README` for details)

### Environment Setup

1. **Configure Environment Variables**  
   Create a `.env` file in the root directory. This file holds your API credentials and overrides default pipeline configurations (using the `GEN_` prefix).
   
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   OPENAI_API_KEY=your_openai_api_key
   
   # Pipeline Overrides
   GEN_RAG_BASE_URL=http://localhost:8000/
   GEN_CHROMA_DB_PATH=your_chroma_db_path
   GEN_CHROMA_COLLECTION_NAME=your_veval_collection_name
   ```

2. **Ensure Directory Permissions**  
   Before starting Docker, ensure the local output directory exists to prevent permission conflicts during volume mounting:
   ```bash
   mkdir -p data
   ```
   
---

### Running the MH-RAG-DIVE Pipeline

**Prerequisite:** Make sure you have $V_{eval}$ set up before proceeding (see `preprocessing/create_veval.ipynb` for details), and that you have specified its path in the `.env` file:

Start the evaluation framework via Docker.

1. **Build the Docker Image**
   ```bash
   docker build -t mh-rag-dive .
   ```

2. **Run a Generation Task**  
   Start a container, mounting your local `.env` and `data/` directory. You can pass CLI arguments directly to override settings.
   
   *Example: Building the Cache using the Fill-Mode*
   ```bash
   docker run --rm -v $(pwd)/data:/app/data --env-file .env mh-rag-dive python src/cli.py --conversations 100 --mode fill
   ```

   *Example: Evaluating a new RAG configuration utilizing the Semantic Cache (Hybrid Mode)*
   ```bash
   docker run --rm -v $(pwd)/data:/app/data --env-file .env mh-rag-dive python src/cli.py --mode hybrid
   ```

#### Execution Modes
- `--mode fill`: Ignores existing cache during generation but builds and saves a new cache tree. Used for the initial bootstrapping run.
- `--mode reuse`: utilizes the existing cache without growing it. Will not cache new follow-up questions if a cache miss occurs.
- `--mode hybrid`: The standard evaluation mode. Loads existing dialogue roots, attempts to use the cache for follow-ups, and falls back to generating (and caching) new questions upon a cache miss.

### Evaluating Conversations (Conversation Evaluator)

Once dialogues are fully generated and saved, run the Conversation Evaluator (CE) locally to score the target RAG system's responses.

1. **Install Dependencies via Poetry:**
   ```bash
   poetry install
   ``` 

2. **Configure Target Files:**
   Open `src/conversation_evaluator/single_turn_evaluation.py` and `src/conversation_evaluator/multi_turn_evaluation.py` in your editor. Define the target dialogue files you wish to evaluate in the configuration block of each script.

3. **Run Single-Turn Evaluation:**
   Evaluates each question individually for correctness, faithfulness, context precision, and context recall.
   ```bash
   poetry run python src/conversation_evaluator/single_turn_evaluation.py
   ```

4. **Run Multi-Turn Evaluation:**
   Evaluates dialogue-level consistency, tracking forgetfulness and context retention.
   ```bash
   poetry run python src/conversation_evaluator/multi_turn_evaluation.py
   ```

*Output:* The scripts process all specified dialogue files and output a new `.json` file in the respective data directory containing both the turn-by-turn individual scores and the aggregated metrics for the run.

---

## Output Artifacts & Data Structure

All generated telemetry, configuration snapshots, and conversation results are stored in the `./data/` directory. Each experiment run generates a timestamped suite of files structured as follows:

### 1. Core Result Files
* 📄 **`dialogues_[TIMESTAMP].jsonl`**: Contains the final, fully validated multi-hop conversations ($N=5$ turns). Each line represents a completed dialogue entry, storing metadata (such as the `root_id`), the sequence of user queries, retrieved contexts from the target RAG system, and the corresponding system answers.
* 📊 **`metrics_[TIMESTAMP].jsonl`**: Provides turn-by-turn telemetry tracking. Each line records a single generation or validation attempt (including rejected attempts and retries). It tracks input/output token counts for each framework component (Discoverer, Generator, Validator), execution latencies, cache hit/miss flags, and validation failure reasons. Every row can be mapped back via its `root_id`.
* ⚙️ **`run_meta_[TIMESTAMP].json`**: A configuration snapshot containing all pipeline parameters, model identifiers, chunk sizes, and API base URLs utilized during that specific run to guarantee full reproducibility.
* 📝 **`pipeline_log_[TIMESTAMP].txt`**: Detailed textual log output capturing the complete execution flow, warnings, and error traces.

### 2. Evaluation Results (`*_evaluation_results_*.json`)
Running the Conversation Evaluator generates dedicated JSON report files containing both raw scores and aggregated statistics:
* **`*_single_turn_evaluation_results.json`**: Turn-level metrics including *Correctness*, *Faithfulness*, *Context Precision*, and *Context Recall*.
* **`*_results_multi_turn.json`** (or multi-turn evaluation files): Dialogue-level metrics tracking longitudinal behavior such as *Forgetfulness* and *Context Retention*.

### 3. Semantic Dialogue Cache (`dialogue_cache/`)
The Cacher component stores persistent state representations inside `data/dialogue_cache/`:
* **`roots/`**: Contains individual `.json` files for each starting point (Turn 0 root dialogue), allowing cross-experiment alignment.
* **`trees/`**: Stores the semantic MCTS-like dialogue trees (`.json`), mapping historical conversational states and embedding vectors to cached follow-up question bundles.

## Metrics Summary

The framework outputs a combination of single-turn and multi-turn RAG metrics

| Metric Type | Metrics |
|-------------|---------|
| **Single-Turn** | Correctness, Faithfulness, Context Precision, Context Recall |
| **Multi-Turn** | Forgetfulness, Context Retention |

## References & Baseline Framework

MH-RAG-DIVE builds upon and extends the foundational architecture of the original RAG-DIVE framework. For details on the baseline methodology and the initial multi-turn evaluation metrics, please refer to the original publication and repository:

* **Repository:** [https://github.com/lorenzbrehme/RAG-DIVE/](https://github.com/lorenzbrehme/RAG-DIVE/)
* **Paper:** Brehme, L., Dornauer, B., Böttcher, J.-H., Schmid, K., Racasan, M.-C., & Breu, R. (2026). RAG-DIVE: A Dynamic Approach for Multi-Turn Dialogue Evaluation in Retrieval-Augmented Generation. *Proceedings of the IEEE/ACM 5th International Conference on AI Engineering - Software Engineering for AI (CAIN ’26)*, 74–84. [DOI: 10.1145/3793653.3793766](http://dx.doi.org/10.1145/3793653.3793766)

```bibtex
@inproceedings{Brehme_2026, 
   series={CAIN ’26},
   title={RAG-DIVE: A Dynamic Approach for Multi-Turn Dialogue Evaluation in Retrieval-Augmented Generation},
   url={[http://dx.doi.org/10.1145/3793653.3793766](http://dx.doi.org/10.1145/3793653.3793766)},
   DOI={10.1145/3793653.3793766},
   booktitle={Proceedings of the IEEE/ACM 5th International Conference on AI Engineering - Software Engineering for AI},
   publisher={ACM},
   author={Brehme, Lorenz and Dornauer, Benedikt and Böttcher, Jan-Henrik and Schmid, Klaus and Racasan, Mircea-Cristian and Breu, Ruth},
   year={2026},
   month=Apr, 
   pages={74–84},
   collection={CAIN ’26} 
}
```