# LLM 3-Phase Architecture

A Python 3.11+ multi-agent system that uses a remote Ollama server, SQLite persistence, a local subject-predicate-object graph, and a FastAPI monitoring surface.

## Agents
- **Repository Agent**: ingestion, chunking, embedding, graph extraction, hybrid retrieval, and context pruning.
- **Diplomat Agent**: user-facing planner using a grounded ReAct-style JSON draft.
- **Judge Agent**: LLM-as-a-Judge verifier that can force retries with targeted feedback.
- **Conductor**: strict JSON handoff orchestrator and monitoring coordinator.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API endpoints
- `GET /health` - liveness check.
- `POST /check` - verifies connectivity to the remote Ollama server.
- `POST /ingest` - ingest a local file into persistent memory with `{"path": "notes.txt"}`.
- `POST /query` - runs the 3-agent workflow with `{"query": "...", "session_id": "optional"}`.
- `GET /status` - service metrics, configured models, and recent workflow runs.
- `GET /runs/{run_id}` - per-run timeline showing which agent is active and what each stage did.

## CLI equivalents
```bash
python main.py check
python main.py ingest notes.txt
python main.py query "What do we know about the customer roadmap?" --session-id demo
python main.py status
python main.py run-status <run_id>
```

## Monitoring
Each query creates a workflow run record plus agent-level events in `database/memory.db`. Use `/status` for a dashboard-friendly overview and `/runs/{run_id}` for a detailed timeline of Repository, Diplomat, Judge, and Conductor activity.

## Persistence
All durable memory is stored in `database/memory.db`, including chunk embeddings, conversation turns, graph triplets, mid-term summaries, workflow runs, and monitoring events.

## Memory augmentation
Semantic memory is persisted in ChromaDB at `database/chroma` by default. After every judge-approved response, the system writes back the user query and final answer embeddings so later queries can retrieve them semantically. Triplets extracted from ingested documents and approved interactions are stored in SQLite and returned alongside vector matches.
